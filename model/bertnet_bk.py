import torch
import torch.nn as nn
import math
import numpy as np
from model.transformer import TransformerBlock
from model.pretext_task import MaskedPrediction, JigsawPrediction_T, ContrastiveLearning, Joint_Prediction
from model.embedding import PositionalEmbedding
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
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=3, num_node=25, num_frame=32,
                 kernel_size=1, stride=1, glo_reg_s=True, att_s=True, glo_reg_t=True, att_t=True,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, use_pes=True, use_pet=True):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                                requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        self.use_temporal_att = use_temporal_att
        if use_temporal_att:
            attt = torch.zeros((1, num_subset, num_frame, num_frame))
            self.register_buffer('attt', attt)
            self.pet = PositionalEncoding(out_channels, num_node, num_frame, 'temporal')
            self.ff_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_t:
                self.in_nett = nn.Conv2d(out_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_t:
                self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame),
                                                requires_grad=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            if use_temporal_att:
                self.downt1 = lambda x: x
            self.downt2 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)  # nctv

            y = self.relu(self.downs1(x) + y)

            y = self.ff_nets(y)

            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)

        if self.use_temporal_att:
            attention = self.attt
            if self.use_pet:
                z = self.pet(y)
            else:
                z = y
            if self.att_t:
                q, k = torch.chunk(self.in_nett(z).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv
                attention = attention + self.tan(
                    torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
            if self.glo_reg_t:
                attention = attention + self.attention0t.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            z = torch.einsum('nctv,nstq->nscqv', [y, attention]).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
            z = self.out_nett(z)  # nctv

            z = self.relu(self.downt1(y) + z)

            z = self.ff_nett(z)

            z = self.relu(self.downt2(y) + z)
        else:
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)

        return z


class BertNet(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset=3, dropout=0., num_person=2,
                 data_channel=3, layers_in_dim=[64], attentiondrop=0, dropout2d=0, out_channel=256, sample_for_pretext_rate=0.2, chose_rate4mask=0.1, mask_divide=[0.8, 0.2],num_seg=3, pretext_task_Train=False, context=False, global_S_A=True, use_dsta=False):
        super(BertNet, self).__init__()

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
        self.global_S_A = global_S_A
        self.use_dsta = use_dsta
        if self.global_S_A == True:
            self.PA = nn.Parameter(torch.zeros(self.num_joint, self.num_joint),requires_grad=True)
            nn.init.constant_(self.PA, 1/self.num_joint)
        self.PE_T = PositionalEmbedding(d_model=self.hiden_dim)
        self.PE_S = nn.Parameter(torch.zeros(self.num_joint, self.hiden_dim))
        nn.init.kaiming_normal_(self.PE_S, a=math.sqrt(5))

        self.input_map = nn.Sequential(
            nn.Conv2d(self.data_channel, layers_in_dim[0], 1),
            nn.BatchNorm2d(layers_in_dim[0]),
            nn.LeakyReLU(0.1),
        )
        if self.use_dsta == True:
            param = {
                'num_node': num_point,
                'num_subset': 3,
                'glo_reg_s': True,
                'att_s': True,
                'glo_reg_t': False,
                'att_t': True,
                'use_spatial_att': True,
                'use_temporal_att': True,
                'attentiondrop': 0,
                'use_pet': False,
                'use_pes': False

            }
            config = [[64, 64, 16, 1], [64, 64, 16, 1], [64, 128, 32, 1], [128, 128, 32, 1], [128, 256, 64, 1],
                      [256, 256, 64, 1], [256, 256, 64, 1], [256, 256, 64, 1]]
            self.encoder = nn.ModuleList()
            num_frame_module = self.len_frames
            for index, (hiden_in, hiden_out, inter_channels, stride) in enumerate(config):
                self.encoder.append(
                    STAttentionBlock(hiden_in, hiden_out, inter_channels, stride=stride, num_frame=num_frame_module,
                                     **param))
                num_frame_module = int(num_frame_module / stride + 0.5)
        else:
            self.encoder = nn.ModuleList(
                [TransformerBlock(hidden_in=self.layers_in_dim[i], hidden_out=self.layers_out_dim[i], attn_heads=self.attn_heads,
                                  feed_forward_hidden=self.layers_out_dim[i], num_joints=num_point, num_frames=self.len_frames,
                                  dropout=dropout, attentiondrop=attentiondrop, context=self.context) for i in range(len(self.layers_in_dim))])
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
        pe_S = self.PE_S.unsqueeze(0).unsqueeze(0)
        pe_T = self.PE_T(x).unsqueeze(2)
        PA = None
        if self.global_S_A == True:
            PA = torch.softmax(self.PA, dim=-1).unsqueeze(0)
        pretext_loss = self.pretext_loss_init
        x_original = x

        x = x.permute(0, 3, 1, 2).contiguous().view(N_x, C, T, V)
        x = self.input_map(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(N_x, T, V, -1)

        x4pretext_task = x[:num4pretext_task]
        x_original4pretext = x_original[:num4pretext_task]

        if self.training == True and self.pretext_task_Train == True:
            x_masked, masks = self.random_mask_all(x4pretext_task)
            N_mask = x_masked.size(0)
            x_jigsaw_T = self.Jigsaw_T_generate(x4pretext_task)
            N_jigsaw_T = x_jigsaw_T.size(0)
            x_predict_S = x4pretext_task
            N_predict_S = x_predict_S.size(0)

            x_predict_S = x_predict_S + pe_T
            x = x + pe_S + pe_T
            x_masked = x_masked + pe_S + pe_T
            x_jigsaw_T = x_jigsaw_T + pe_S + pe_T

            # N_all = [N_x, N_mask, N_jigsaw_T, N_predict_S]
            x_all = torch.cat([x, x_masked, x_jigsaw_T, x_predict_S], dim=0)
            if self.use_dsta == True:
                x_all = x_all.permute(0, 3, 1, 2).contiguous()
                for transformer in self.encoder:
                    x_all = transformer.forward(x_all)
                x_all = x_all.permute(0, 2, 3, 1)
            else:
                for transformer in self.encoder:
                    x_all = transformer.forward(x_all, R_A=PA)

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
            x = x + pe_S + pe_T
            if self.use_dsta == True:
                x = x.permute(0, 3, 1, 2).contiguous()
                for transformer in self.encoder:
                    x = transformer.forward(x)
                x = x.permute(0, 2, 3, 1)
            else:
                for transformer in self.encoder:
                    x = transformer.forward(x, R_A=PA)

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
