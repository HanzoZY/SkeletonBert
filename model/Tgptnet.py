import torch
import torch.nn as nn
import math
import numpy as np
from ipdb import set_trace
from model.pretext_task import MaskedPrediction, JigsawPrediction_T, ContrastiveLearning_SimSiam, Joint_Prediction
import itertools
EPS = 1e-8


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


def get_mask_array(size_in, divide_rate= [0.8,0.1,0.1]):
    '''
    generate mask for each subset masks of batch data, masks[0] represents the index of unmasked samples,

    Input:
         size_in: the length of mask array
         divide_rate: the first: the rate of remain; the second: the rate of set empty(0); the third: the rate of noise
    Returns:
         N masks, each mask has the size :

    '''
    chosen_list = []
    for i in divide_rate:
        chosen_list.append(int(size_in*i))
    new_array = np.zeros(size_in)
    flag = 0
    for idx, num in enumerate(chosen_list):
        new_array[flag:flag+num] = idx
        flag = flag+num
    np.random.shuffle(new_array)
    map_clip = []
    for idx in range(len(divide_rate)):
         map_clip.append((new_array==idx).astype(int))

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


class PositionalEmbedding(nn.Module):
    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEmbedding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len
        self.domain = domain
        if domain == "temporal":
            self.PE = nn.Parameter(torch.zeros(channel, self.time_len))
        elif domain == "spatial":
            self.PE = nn.Parameter(torch.zeros(channel, self.joint_num))
        # nn.init.kaiming_uniform_(self.PE)
        nn.init.uniform_(self.PE)

    def forward(self, x):  # nctv
        if self.domain == "spatial":
            pe = self.PE.unsqueeze(1).unsqueeze(0)
        else:
            pe = self.PE.unsqueeze(-1).unsqueeze(0)
        x = x + pe
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

        backward_mask = torch.triu(torch.ones(num_frame, num_frame))
        self.register_buffer('backward_mask', backward_mask)

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
                self.in_nett = nn.Conv2d(out_channels, 6 * num_subset * inter_channels, 1, bias=True)
                # self.alphat = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
                self.alphat_0 = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
                self.alphat_1 = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
                self.alphat_2 = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_t:
                self.attention0t = nn.Parameter(torch.zeros(1, num_subset, num_frame, num_frame) + torch.eye(num_frame),
                                                requires_grad=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * num_subset * 3, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.out_nett_extend = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels), )
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
            self.downt3 = nn.Sequential(
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
            self.downt3 = lambda x: x

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

        # set_trace()
        # y_1 = self.out_nett_extend(y)
        # y_1 = self.relu(self.downt3(y) + y_1)
        # y = y_1

        forward_mask = self.backward_mask.transpose(-1, -2)
        backward_mask = self.backward_mask
        if self.use_temporal_att:
            attention = self.attt
            if self.use_pet:
                z = self.pet(y)
            else:
                z = y
            q_k_in = self.in_nett(z).view(N, 6 * self.num_subset, self.inter_channels, T, V)
            q_f, q_b, q_c, k_f, k_b, k_c = torch.chunk(q_k_in, 6, dim=1)
            attention_b = torch.einsum('nsctv,nscqv->nstq', [q_b, k_b]) / (self.inter_channels * V) * self.alphat_0
            attention_f = torch.einsum('nsctv,nscqv->nstq', [q_f, k_f]) / (self.inter_channels * V) * self.alphat_1
            attention_c = torch.einsum('nsctv,nscqv->nstq', [q_c, k_c]) / (self.inter_channels * V) * self.alphat_2
            attention_b = torch.einsum('nstq,tq->nstq', [attention_b, backward_mask])
            attention_f = torch.einsum('nstq,tq->nstq', [attention_f, forward_mask])
            attention_b = self.drop(attention_b)
            attention_f = self.drop(attention_f)
            attention_c = self.drop(attention_c)
            z_f = torch.einsum('nctv,nstq->nscqv', [y, attention_f]).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
            z_b = torch.einsum('nctv,nstq->nscqv', [y, attention_b]).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
            z_c = torch.einsum('nctv,nstq->nscqv', [y, attention_c]).contiguous() \
                .view(N, self.num_subset * self.out_channels, T, V)
            z = torch.cat([z_f, z_b, z_c], dim=-3)
            z = self.out_nett(z)  # nctv

            z = self.relu(self.downt1(y) + z)

            z = self.ff_nett(z)

            z = self.relu(self.downt2(y) + z)
        else:
            z = self.out_nett(y)
            z = self.relu(self.downt2(y) + z)

        # set_trace()
        z_1 = self.out_nett_extend(z)
        z_1 = self.relu(self.downt3(z) + z_1)
        z = z_1
        return z


class GPTNet(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset=3, dropout=0., config=None, num_person=2,
                 num_channel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=True, mask_divide=[0.8, 0.1, 0.1],
                 var=0.15, use_SSL=False, num_seg=3,
                 use_temporal_att=True, use_spatial_att=True, attentiondrop=0, dropout2d=0, use_pet=True, use_pes=True,
                 SSL_weight={'mask':0.1, 'pred_S':0.1, 'pred_T':0.1, 'Contra':0.1}):
        super(GPTNet, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        self.init_loss()
        self.var = var
        self.use_SSL = use_SSL
        self.SSL_weight = SSL_weight
        self.mask_divide = mask_divide
        self.num_seg = num_seg


        self.num_person = num_person
        self.num_point = num_point
        # self.pes = PositionalEncoding(in_channels, num_point, num_frame, 'spatial')
        self.pet = PositionalEncoding(in_channels, num_point, num_frame, 'temporal')
        self.pes = PositionalEmbedding(in_channels, num_point, num_frame, 'spatial')
        # self.pet = PositionalEmbedding(in_channels, num_point, num_frame, 'temporal')
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        param = {
            'num_node': num_point,
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)

        self.fc = nn.Linear(self.out_channels, num_class)
        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)
        self.init_jigsaw()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

        self.SSL_mask = MaskedPrediction(hidden=self.out_channels, num_person=self.num_person, reconstruct=num_channel)
        self.SSL_JigsawT = JigsawPrediction_T(hid_dim=self.out_channels, num_perm=len(self.permutations_T))
        self.SSL_JointP = Joint_Prediction(hid_dim=self.out_channels, num_joints=self.num_point)
        self.SSL_Contra = ContrastiveLearning_SimSiam(hid_dim=self.out_channels)


    def init_loss(self):
        pretext_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('pretext_loss_init', pretext_loss_init)
        mask_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('mask_loss_init', mask_loss_init)
        joint_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('joint_loss_init', joint_loss_init)
        simloss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('simloss_init', simloss_init)
        jigsaw_T_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('jigsaw_T_loss_init', jigsaw_T_loss_init)





    def init_jigsaw(self):
        # initialize permutation for temporal dimensio
        temp_list_T = list(range(self.num_seg))
        self.permutations_T = list(itertools.permutations(temp_list_T))



    def Jigsaw_T_generate(self,x):
        N, C, T, V, M = x.shape
        idx = list(range(T))
        cut_num = int(T/self.num_seg)
        cut_idx = np.array([idx[i*cut_num:(i+1)*cut_num] if i < self.num_seg-1 else idx[i*cut_num:] for i in range(self.num_seg)])
        x_list = []
        num_perm = len(self.permutations_T)

        for i, idx_chose in enumerate(self.permutations_T):
            idx_i_permute = cut_idx[list(idx_chose)].tolist()
            idx_i = [j for k in idx_i_permute for j in k]
            x_list.append(x[:,:,idx_i,:,:])

        x_T = torch.stack(x_list)
        assert num_perm == x_T.size(0)
        x_T = x_T.view(num_perm, N, C, T, V, M)

        return x_T


    def random_mask_all(self,x):
        '''
        all samples share the same mask and noise
        Args:
            x: all input data

        Returns:

        '''
        N, C, T, V, M = x.shape
        noise = torch.FloatTensor(*x.size()[1:-1]).uniform_(-self.var, self.var).unsqueeze(0).unsqueeze(-1)
        masks = torch.tensor(data=get_mask_array(size_in=V*T, divide_rate=self.mask_divide),dtype=torch.float)

        # noise = noise.to(x.get_device())
        # masks = masks.to(x.get_device())

        x_masked = x
        num_mask = masks.size(0)
        set_trace()
        assert num_mask==len(self.mask_divide)
        for i, mask in enumerate(masks.chunk(num_mask, dim=0)):
            if i==0:
                continue
            if i==1:
                mask = 1-mask.view(T,V).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                x_masked = x_masked * mask
            if i==2:
                mask_noise = mask.view(T,V).unsqueeze(0).unsqueeze(0).unsqueeze(-1) * noise
                x_masked = x_masked + mask_noise

        return x_masked

    def forward(self, x):
        """

        :param x: N M C T V
        :return: classes scores
        """
        N, C, T, V, M = x.shape
        pretext_loss = self.pretext_loss_init.expand(N)
        mask_loss = self.mask_loss_init.expand(N)
        jigsaw_T_loss = self.jigsaw_T_loss_init.expand(N)
        joint_loss = self.joint_loss_init.expand(N)
        simloss = self.simloss_init.expand(N)




        # set_trace()
        # if self.use_SSL == True and self.training == True:
        if self.use_SSL == True:
            x_jigsaw_T = self.Jigsaw_T_generate(x=x).detach()
            num_perm = x_jigsaw_T.size(0)
            assert num_perm == len(self.permutations_T)
            x_jigsaw_T = x_jigsaw_T.view(num_perm, N, C, T, V, M)
            x_masked = self.random_mask_all(x=x).detach()
            set_trace()
            x_origin = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
            x_masked = x_masked.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
            x_jigsaw_T = x_jigsaw_T.permute(0, 1, 5, 2, 3, 4).contiguous().view(num_perm * N * M, C, T, V)

            x = x_origin

            x_temp = self.input_map(x)
            x_masked = self.input_map(x_masked)
            x_jigsaw_T = self.input_map(x_jigsaw_T)
            x = self.pet(self.pes(x_temp))
            x_masked = self.pet(self.pes(x_masked))
            x_jigsaw_T = self.pet(self.pes(x_jigsaw_T))
            x_predict_S = self.pet(x_temp)
            for i, m in enumerate(self.graph_layers):
                x = m(x)
                x_masked = m(x_masked)
                x_jigsaw_T = m(x_jigsaw_T)
                x_predict_S = m(x_predict_S)

            # NM, C, T, V

            '''predict the masked coordinates'''
            mask_loss = self.SSL_mask(x_origin=x_origin, x_masked=x_masked)
            x_masked = x_masked.view(N, M, self.out_channels, T, V)
            x_masked = x_masked.mean(-1).mean(-1).mean(1).view(N,self.out_channels).unsqueeze(0)


            '''predict the Jigsaw T'''
            x_jigsaw_T = x_jigsaw_T.view(num_perm, N, M, self.out_channels, T, V)
            x_jigsaw_T = x_jigsaw_T.mean(-1).mean(-1).mean(2).view(num_perm, N, self.out_channels)
            jigsaw_T_loss = self.SSL_JigsawT(x=x_jigsaw_T)



            '''predict the joint type'''
            x_predict_S = x_predict_S.view(N, M, self.out_channels, T, V)
            x_predict_S = x_predict_S.mean(-2).mean(1).view(N, self.out_channels, V)
            joint_loss = self.SSL_JointP(x=x_predict_S)
            x_predict_S = x_predict_S.mean(-1).view(N, self.out_channels).unsqueeze(0)


            ''' ContrastiveLearning '''
            x = x.view(N, M, self.out_channels, T*V)
            x4SSL = x.mean(-1).mean(1).view(N, self.out_channels).unsqueeze(0)
            x4SSL = torch.cat([x4SSL,x_masked,x_jigsaw_T,x_predict_S],dim=0)
            simloss = self.SSL_Contra(x4SSL)



            ''' downstream task(recognition)'''
            x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
            x = self.drop_out2d(x)
            x = x.mean(3).mean(1)

            x = self.drop_out(x)  # whole spatial of one channel

            pretext_loss = pretext_loss + self.SSL_weight['mask'] * mask_loss + self.SSL_weight['pred_S'] * joint_loss + \
                           self.SSL_weight['pred_T'] * jigsaw_T_loss + self.SSL_weight['Contra'] * simloss
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
            x = self.input_map(x)
            x = self.pes(x)
            x = self.pet(x)

            for i, m in enumerate(self.graph_layers):
                x = m(x)

            # NM, C, T, V

            x = x.view(N, M, self.out_channels, -1)
            x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
            x = self.drop_out2d(x)
            x = x.mean(3).mean(1)

            x = self.drop_out(x)  # whole spatial of one channel



        return self.fc(x), pretext_loss, mask_loss, jigsaw_T_loss, joint_loss, simloss


if __name__ == '__main__':
    # config = [[64, 64, 16, 1], [64, 64, 16, 1],
    #           [64, 128, 32, 2], [128, 128, 32, 1],
    #           [128, 256, 64, 2], [256, 256, 64, 1],
    #           [256, 256, 64, 1], [256, 256, 64, 1],
    #           ]
    # net = GPTNet(config=config)  # .cuda()
    # ske = torch.rand([2, 3, 32, 25, 2])  # .cuda()
    # print(net(ske).shape)
    get_mask_array(20)

