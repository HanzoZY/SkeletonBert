import torch
import torch.nn as nn
import math
import numpy as np
from model.pretext_task import MaskedPrediction, JigsawPrediction_T, ContrastiveLearning, Joint_Prediction
import itertools

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)
def get_mask_array(size_in, rate_chose=0.2, chose_divide= [0.8,0.2]):
    '''
    generate mask for each subset masks of batch data, masks[0] represents the index of unmasked samples

    '''
    num_chosen = int(size_in * rate_chose)
    num_unchosen = size_in - num_chosen
    chosen_list = [num_unchosen]
    for i in chose_divide:
        chosen_list.append(int(num_chosen*i))
    new_array = np.zeros(size_in)
    flag = 0
    for idx, num in enumerate(chosen_list):
        new_array[flag:flag+num] = idx
        flag = flag+num
    np.random.shuffle(new_array)
    map_clip = [(new_array==0).astype(int)]
    for idx in range(len(chose_divide)):
         map_clip.append((new_array==(idx+1)).astype(int))

    return np.stack(map_clip)

class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        # pe = position/position.max()*2 -1
        # pe = pe.view(time_len, joint_num).unsqueeze(0).unsqueeze(0)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # ntvc
        x = x + self.pe[:, :x.size(1)]
        return x

class PositionalEmbedding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEmbedding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            self.PE = nn.Parameter(torch.zeros(self.time_len,channel))
        elif domain == "spatial":
            # spatial embedding
            self.PE = nn.Parameter(torch.zeros(self.joint_num,channel))
        # nn.init.kaiming_uniform_(self.PE)
        nn.init.uniform_(self.PE)


    def forward(self, x):  # ntvc
        if self.domain == "spatial":
            pe = self.PE.unsqueeze(0).unsqueeze(0)
        else:
            pe = self.PE.unsqueeze(1).unsqueeze(0)
        x = x + pe
        return x
class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=32,
                 kernel_size=1, glo_reg_s=True, att_s=True, layernorm_affine=False, context=True,
                 use_TCN=False, attentiondrop=0):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.layernorm_affine = layernorm_affine
        self.use_TCN = use_TCN
        self.context = context
        self.drop = nn.Dropout(attentiondrop)
        atts = torch.zeros((1, num_subset, 1,num_node, num_node))
        self.register_buffer('atts', atts)

        backward_mask = torch.triu(torch.ones(num_frame,num_frame))
        self.register_buffer('backward_mask', backward_mask)


        self.ff_nets = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels, elementwise_affine=self.layernorm_affine)
        )
        if att_s:
            self.in_nets = nn.Linear(in_channels, 2 * num_subset * inter_channels)
        if glo_reg_s:
            self.attention0s = nn.Parameter(torch.ones(num_subset, num_node, num_node) / num_node,
                                            requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Linear(in_channels * num_subset, out_channels),
            nn.LayerNorm(out_channels, elementwise_affine=self.layernorm_affine)
        )
        # temporal modules
        self.ff_nett = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels,elementwise_affine=self.layernorm_affine)
        )
        self.in_nett_f = nn.Linear(out_channels, 2 * num_subset * inter_channels)
        self.in_nett_b = nn.Linear(out_channels, 2 * num_subset * inter_channels)
        self.out_nett_f = nn.Sequential(
            nn.Linear(out_channels * num_subset, out_channels),
            nn.LayerNorm(out_channels,elementwise_affine=self.layernorm_affine)
        )
        self.out_nett_b = nn.Sequential(
            nn.Linear(out_channels * num_subset, out_channels),
            nn.LayerNorm(out_channels,elementwise_affine=self.layernorm_affine)
        )
        if self.use_TCN:
            self.TCN = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(3, 0), bias=True, stride=(1, 1)),
                nn.BatchNorm2d(out_channels)
            )

        if in_channels != out_channels:
            self.downs1 = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels,elementwise_affine=self.layernorm_affine),
            )
            self.downs2 = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels,elementwise_affine=self.layernorm_affine),
            )
            self.downt1 = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.LayerNorm(out_channels,elementwise_affine=self.layernorm_affine),
            )
            self.downt2 = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.LayerNorm(out_channels,elementwise_affine=self.layernorm_affine),
            )
            if self.use_TCN:
                self.down_tcn = nn.Sequential(
                    nn.Linear(out_channels, out_channels),
                    nn.LayerNorm(out_channels, elementwise_affine=self.layernorm_affine)
                )
        else:
            self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt1 = lambda x: x
            self.downt2 = lambda x: x
            if self.use_TCN:
                self.down_tcn = lambda x: x


        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, T, V, C = x.size()
        attention_s = self.atts
        y = x
        # spatial operation
        if self.att_s:
            q_k_in_s = self.elu(self.in_nets(y).view(N, T, V, self.inter_channels, 2 * self.num_subset))+1
            q_s, k_s = torch.chunk(q_k_in_s, 2, dim=-1)  # q,k shape: n,t,v,c,num_sub
            # attention = attention + self.tan(
            #     torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.context:
                attention_context = torch.einsum('ntvcs,ntucs->nsvu', [q_s, k_s]) / (self.inter_channels * T)
                attention_context = attention_context/attention_context.sum(-1, keepdim=True)
                attention_context = attention_context.unsqueeze(2).expand(-1, -1, T, -1, -1)
                attention_s = attention_s + attention_context
            else:
                attention_noncontext = torch.einsum('ntvcs,ntucs->nstvu', [q_s, k_s]) / (self.inter_channels)
                attention_noncontext = attention_noncontext/attention_noncontext.sum(-1, keepdim=True)
                attention_s = attention_s + attention_noncontext
        if self.glo_reg_s:
            attention0s = self.attention0s.unsqueeze(0).unsqueeze(2)
            attention_s = attention_s + attention0s.expand(N, -1, T, -1, -1)
            # attention = attention + self.attention0s.repeat(N, 1, 1, 1)
        attention_s = self.drop(attention_s)
        N_s, S_s, T_s, V_s_0, V_s_1 = attention_s.size()
        assert N_s==N and S_s==self.num_subset and T_s==T and V_s_0==V_s_1==V
        y = torch.einsum('nstvu,ntuc->ntvcs', [attention_s,x]).contiguous().view(N, T, V, self.num_subset * self.in_channels)
        y = self.out_nets(y)  # nctv
        y = self.gelu(self.downs1(x) + y)
        y = self.ff_nets(y)
        y = self.gelu(self.downs2(x) + y)

        #    temporal operation
        forward_mask = self.backward_mask.transpose(-1, -2)
        backward_mask = self.backward_mask
        z_f = y
        z_b = y
        q_k_in_f = self.elu(self.in_nett_f(z_f).view(N, T, V, self.inter_channels, 2 * self.num_subset)) + 1
        q_k_in_b = self.elu(self.in_nett_b(z_b).view(N, T, V, self.inter_channels, 2 * self.num_subset)) + 1
        q_f, k_f = torch.chunk(q_k_in_f, 2, dim=-1)
        q_b, k_b = torch.chunk(q_k_in_b, 2, dim=-1)
        if self.context:
            attention_context_f = torch.einsum('ntvcs,nkvcs->nstk', [q_f, k_f]) / (self.inter_channels * T)
            attention_context_b = torch.einsum('ntvcs,nkvcs->nstk', [q_b, k_b]) / (self.inter_channels * T)
            attention_context_f = torch.einsum('nstk,tk->nstk', [attention_context_f, forward_mask])
            attention_context_b = torch.einsum('nstk,tk->nstk', [attention_context_b, backward_mask])
            attention_context_f = attention_context_f / attention_context_f.sum(-1, keepdim=True)
            attention_context_b = attention_context_b / attention_context_b.sum(-1, keepdim=True)
            attention_context_f = attention_context_f.unsqueeze(2).expand(-1, -1, V, -1, -1)
            attention_context_b = attention_context_b.unsqueeze(2).expand(-1, -1, V, -1, -1)
            attention_f = self.drop(attention_context_f)
            attention_b = self.drop(attention_context_b)
        else:
            attention_noncontext_f = torch.einsum('ntvcs,nkvcs->nsvtk', [q_f, k_f]) / (self.inter_channels)
            attention_noncontext_b = torch.einsum('ntvcs,nkvcs->nsvtk', [q_b, k_b]) / (self.inter_channels)
            attention_noncontext_f = torch.einsum('nsvtk,tk->nsvtk', [attention_noncontext_f, forward_mask])
            attention_noncontext_b = torch.einsum('nsvtk,tk->nsvtk', [attention_noncontext_b, backward_mask])
            attention_noncontext_f = attention_noncontext_f / attention_noncontext_f.sum(-1, keepdim=True)
            attention_noncontext_b = attention_noncontext_b / attention_noncontext_b.sum(-1, keepdim=True)

            attention_f = self.drop(attention_noncontext_f)
            attention_b = self.drop(attention_noncontext_b)

        N_a_f, S_a_f, V_a_f, T_a_0_f, T_a_1_f = attention_f.size()
        N_a_b, S_a_b, V_a_b, T_a_0_b, T_a_1_b = attention_b.size()
        assert N_a_f==N_a_b==N and S_a_f==S_a_b==self.num_subset and V_a_b==V_a_f==V and T_a_0_f==T_a_1_f==T_a_0_b==T_a_1_b==T
        z_f = torch.einsum('nsvtk,nkvc->ntvcs', [attention_f, y]).contiguous().view(N, T, V,
                                                                                     self.num_subset * self.out_channels)
        z_b = torch.einsum('nsvtk,nkvc->ntvcs', [attention_b, y]).contiguous().view(N, T, V,
                                                                                     self.num_subset * self.out_channels)
        z_f = self.out_nett_f(z_f) # ntvc
        z_b = self.out_nett_b(z_b)
        z = self.gelu(self.downt1(y) + z_f + z_b)
        z = self.ff_nett(z)
        z = self.gelu(self.downt2(y) + z)
        if self.use_TCN == True:
            z_t = self.TCN(z)
            z_t = self.gelu(self.down_tcn(z) + z_t)
            z = z_t
        return z


class GPTNet(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset=3, dropout=0., num_person=2, glo_reg_s=True,
                 use_TCN=False, layernorm_affine=False, att_s=True, kernel_size=7,
                 data_channel=3, layers_in_dim=[64], attentiondrop=0, dropout2d=0, out_channel=256,
                 sample_for_pretext_rate=0.2, chose_rate4mask=0.1, mask_divide=[0.8, 0.2], num_seg=3,
                 pretext_task_Train=False, context=False):
        super(GPTNet, self).__init__()

        self.out_channels = out_channel
        self.data_channel = data_channel
        self.attn_heads = num_subset
        self.len_frames = num_frame
        self.num_joint = num_point
        self.chose_rate = chose_rate4mask
        self.layers_in_dim = layers_in_dim
        self.layers_out_dim = self.layers_in_dim[1:].copy()
        self.layers_out_dim.append(out_channel)
        self.hiden_dim = self.layers_in_dim[0]
        self.mask_divide = mask_divide
        self.sample4pretext_rate = sample_for_pretext_rate
        self.num_seg = num_seg
        self.context = context
        self.pretext_task_Train = pretext_task_Train

        self.PE_T = PositionalEncoding(self.hiden_dim, self.num_joint, num_frame, 'temporal')
        self.PE_S = PositionalEmbedding(self.hiden_dim, self.num_joint, num_frame, 'spatial')

        # self.input_map = nn.Sequential(
        #     nn.Conv2d(self.data_channel, layers_in_dim[0], 1),
        #     nn.BatchNorm2d(layers_in_dim[0]),
        #     nn.LeakyReLU(0.1),
        # )
        self.input_map = nn.Sequential(
            nn.Linear(self.data_channel, layers_in_dim[0]),
            nn.LayerNorm(layers_in_dim[0],elementwise_affine=layernorm_affine),
            nn.LeakyReLU(0.1),
        )
        param = {
            'num_node': num_point,
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'attentiondrop': attentiondrop,
            'use_TCN': use_TCN,
            'layernorm_affine': layernorm_affine,
            'kernel_size': kernel_size,
            'context': context
        }
        self.encoder = nn.ModuleList([STAttentionBlock(in_channels=self.layers_in_dim[i],
                                                       out_channels=self.layers_out_dim[i],
                                                       inter_channels=int(self.layers_out_dim[i] / 4),
                                                       num_frame=self.len_frames, **param) for i in
                                      range(len(self.layers_in_dim))])
        self.init_jigsaw()
        self.mask_pretext_task = MaskedPrediction(hidden=self.out_channels, reconstruct=self.data_channel)
        self.jigsaw_T_pretext_task = JigsawPrediction_T(hidden=self.out_channels, num_perm=len(self.permutations_T))
        self.constrastive_pretext_task = ContrastiveLearning(num_perm=len(self.permutations_T))
        self.joints_pretext_task = Joint_Prediction(hidden=self.out_channels, num_joints=self.num_joint)

        self.fc = nn.Linear(self.out_channels, num_class)

        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
        pretext_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('pretext_loss_init', pretext_loss_init)

    def random_mask_all(self,x):
        N, T, V, C = x.size()
        # mask = torch.tensor(data=get_mask_array(size_in=V*T, rate_chose=self.chose_rate, chose_divide=self.mask_divide),dtype=torch.float).to(x.get_device())
        mask = torch.tensor(data=get_mask_array(size_in=V*T, rate_chose=self.chose_rate, chose_divide=self.mask_divide),dtype=torch.float)
        mask = mask.unsqueeze(1).unsqueeze(-1)
        mask = mask.view(len(self.chose_divide)+1, 1, T, V, 1)
        x_masked = x * mask[0]
        return x_masked, mask


    def init_jigsaw(self):
        # initialize permutation for temporal dimensio
        temp_list_T = list(range(self.num_seg))
        self.permutations_T = list(itertools.permutations(temp_list_T))

    def Jigsaw_T_generate(self,x):
        N, T, V, C = x.size()
        idx = list(range(T))
        cut_num = int(T/self.num_seg)
        cut_idx = np.array([idx[i*cut_num:(i+1)*cut_num] if i < self.num_seg-1 else idx[i*cut_num:] for i in range(self.num_seg)])
        x_list = []
        num_perm = len(self.permutations_T)
        for i, idx_chose in enumerate(self.permutations_T):
            idx_i_permute = cut_idx[list(idx_chose)].tolist()
            idx_i = [j for k in idx_i_permute for j in k]
            x_list.append(x[:,idx_i,:,:])

        x_T = torch.stack(x_list)
        assert num_perm == x_T.size(0)
        x_T = x_T.view(num_perm*N , T, V, C)

        return x_T


    def forward(self, x):
        """

        :param x: N M C T V
        :return: classes scores
        """
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
        N_x = x.size(0)
        num4pretext_task = int(self.sample4pretext_rate * N_x)
        pretext_loss = self.pretext_loss_init
        x_original = x

        # x = x.permute(0, 3, 1, 2).contiguous().view(N_x, C, T, V)
        # x = self.input_map(x)
        # x = x.permute(0, 2, 3, 1).contiguous().view(N_x, T, V, -1) # x shape: N,T,V,C

        x = self.input_map(x)

        x4pretext_task = x[:num4pretext_task]
        x_original4pretext = x_original[:num4pretext_task]

        if self.training == True and self.pretext_task_Train == True:

            x_masked, masks = self.random_mask_all(x4pretext_task)
            N_mask = x_masked.size(0)
            x_jigsaw_T = self.Jigsaw_T_generate(x4pretext_task)
            N_jigsaw_T = x_jigsaw_T.size(0)
            x_predict_S = x4pretext_task
            N_predict_S = x_predict_S.size(0)

            x_predict_S = self.PE_T(x_predict_S)
            x = self.PE_S(self.PE_T(x))
            x_masked = self.PE_S(self.PE_T(x_masked))
            x_jigsaw_T = self.PE_S(self.PE_T(x_jigsaw_T))

            # N_all = [N_x, N_mask, N_jigsaw_T, N_predict_S]
            x_all = torch.cat([x, x_masked, x_jigsaw_T, x_predict_S], dim=0)
            for transformer in self.encoder:
                x_all = transformer.forward(x_all)

            count = 0
            x = x_all[count: count + N_x]
            count = count + N_x
            x_masked = x_all[count: count + N_mask]
            count = count + N_mask
            x_jigsaw_T = x_all[count: count + N_jigsaw_T]
            count = count + N_jigsaw_T
            x_predict_S = x_all[count: count + N_predict_S]
            count = count + N_predict_S
            assert count == x_all.size(0)
            mask_loss = self.mask_pretext_task(x_origin=x_original4pretext, x_masked=x_masked, masks=masks)
            Jigsaw_loss_T = self.jigsaw_T_pretext_task(x_jigsaw_T)
            Predict_joints_loss = self.joints_pretext_task(x_predict_S)
            Contrastive_loss = self.constrastive_pretext_task(x=x[:num4pretext_task], x_masked=x_masked,
                                                              x_jigsaw_T=x_jigsaw_T, x_predict_S=x_predict_S)
            pretext_loss = mask_loss + pretext_loss + Jigsaw_loss_T + Predict_joints_loss + Contrastive_loss
        else:
            x = self.PE_S(x)
            x = self.PE_T(x)
            for transformer in self.encoder:
                x = transformer.forward(x)

        # NM, C, T, V
        x = x.permute(0, 3, 1, 2).contiguous().view(N_x, self.out_channels, T, V)
        x = x.view(N, M, self.out_channels, -1)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
        x = self.drop_out2d(x)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)  # whole spatial of one channel

        return self.fc(x), pretext_loss.unsqueeze(0).expand(N, -1)


if __name__ == '__main__':
    pass
    # config = [[64, 64, 16, 1], [64, 64, 16, 1],
    #           [64, 128, 32, 2], [128, 128, 32, 1],
    #           [128, 256, 64, 2], [256, 256, 64, 1],
    #           [256, 256, 64, 1], [256, 256, 64, 1],
    #           ]
    # net = DSTANet(config=config)  # .cuda()
    # ske = torch.rand([2, 3, 32, 25, 2])  # .cuda()
    # print(net(ske).shape)
