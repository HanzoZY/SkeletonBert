import torch
import torch.nn as nn
import math
import numpy as np
from ipdb import set_trace
from model.pretext_task_option import MaskedPrediction, ContrastiveLearning_SimSiam, Joint_Prediction, ReversePrediction, JigsawPrediction_T_labeled
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


# def get_mask_array(size_in, divide_rate= [0.8,0.1,0.1]):
#     '''
#     generate mask for each subset masks of batch data, masks[0] represents the index of unmasked samples,
#
#     Input:
#          size_in: the length of mask array
#          divide_rate: the first: the rate of remain; the second: the rate of set empty(0); the third: the rate of noise
#     Returns:
#          N masks, each mask has the size :
#
#     '''
#     chosen_list = []
#     for i in divide_rate:
#         chosen_list.append(int(size_in*i))
#     new_array = np.zeros(size_in)
#     flag = 0
#     for idx, num in enumerate(chosen_list):
#         new_array[flag:flag+num] = idx
#         flag = flag+num
#     np.random.shuffle(new_array)
#     map_clip = []
#     for idx in range(len(divide_rate)):
#          map_clip.append((new_array==idx).astype(int))
#
#     return np.stack(map_clip)

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
        pe = self.pe[:, :, :x.size(2)]
        return pe


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
        return pe




class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset_S=3, num_subset_T=2, num_node=25,
                 num_frame=32, parallel=False, S_atten='free', T_atten='context',
                 kernel_size=1, stride=1, glo_reg_s=True, directed=True, TCN=True,
                 attentiondrop=0):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset_S = num_subset_S
        self.num_subset_T = num_subset_T
        self.glo_reg_s = glo_reg_s
        self.directed = directed
        self.TCN = TCN
        self.parallel = parallel
        self.S_atten = S_atten
        self.T_atten = T_atten

        backward_mask = torch.triu(torch.ones(num_frame, num_frame))
        self.register_buffer('backward_mask', backward_mask)

        pad = int((kernel_size - 1) / 2)

        '''S-TR'''
        atts = torch.zeros((1, self.num_subset_S, num_node, num_node))
        self.alphas = nn.Parameter(torch.ones(1, self.num_subset_S, 1, 1), requires_grad=True)
        if glo_reg_s:
            self.attention0s = nn.Parameter(torch.ones(1, self.num_subset_S, num_node, num_node) / num_node,
                                            requires_grad=True)
        self.register_buffer('atts', atts)
        self.in_nets = nn.Conv2d(in_channels, 2 * self.num_subset_S * inter_channels, 1, bias=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * self.num_subset_S, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.ff_nets = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        '''T-TR'''
        if self.directed == True:
            self.alphat_b = nn.Parameter(torch.ones(1, self.num_subset_T, 1, 1), requires_grad=True)
            self.alphat_f = nn.Parameter(torch.ones(1, self.num_subset_T, 1, 1), requires_grad=True)
        else:
            self.alphat = nn.Parameter(torch.ones(1, 2 * self.num_subset_T, 1, 1), requires_grad=True)
            num_subset_T_un = 2 * self.num_subset_T

        if self.parallel == True:
            self.in_nett = nn.Conv2d(in_channels, 4 * self.num_subset_T * inter_channels, 1, bias=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(in_channels * self.num_subset_T * 2, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.in_channels_t = in_channels
        else:
            self.in_nett = nn.Conv2d(out_channels, 4 * self.num_subset_T * inter_channels, 1, bias=True)
            self.out_nett = nn.Sequential(
                nn.Conv2d(out_channels * self.num_subset_T * 2, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.in_channels_t = out_channels

        self.ff_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
            nn.BatchNorm2d(out_channels),
        )


        if self.TCN==True:
            self.out_nett_extend = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels), )
        if in_channels != out_channels or stride != 1:
            self.downs1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )

            self.downt1 = nn.Sequential(
                nn.Conv2d(self.in_channels_t, out_channels, 1, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downt2 = nn.Sequential(
                nn.Conv2d(self.in_channels_t, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if self.TCN==True:
                self.downtTCN = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                    nn.BatchNorm2d(out_channels),
                )
        else:
            self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt1 = lambda x: x
            self.downt2 = lambda x: x
            if self.TCN==True:
                self.downtTCN = lambda x: x

        # self.ff_net_all = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
        #     nn.BatchNorm2d(out_channels),
        # )

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        N, C, T, V = x.size()

        attention = self.atts
        if self.glo_reg_s:
            attention = attention + self.attention0s

        if self.S_atten == 'free':
            attention = attention.unsqueeze(2)
            alphas = self.alphas.unsqueeze(2)
        else:
            alphas = self.alphas


        y = x
        if self.S_atten == 'context_new':
            q, k = torch.chunk(self.in_nets(torch.mean(input=y,dim=-2,keepdim=True)).view(N, 2 * self.num_subset_S, self.inter_channels, V), 2,
                               dim=1)  # nctv -> n num_subset c'tv
        else:
            q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset_S, self.inter_channels, T, V), 2,
                           dim=1)  # nctv -> n num_subset c'tv
        if self.S_atten == 'context' or self.S_atten == 'context_new':
            if self.S_atten == 'context':
                q = q.mean(-2)
                k = k.mean(-2)
            attention = attention + self.tan(
                torch.einsum('nscu,nscv->nsuv', [q, k]) / (self.inter_channels)) * alphas
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous().view(N,
                                                                                   self.num_subset_S * self.in_channels,
                                                                                   T, V)
        elif self.S_atten == 'avg':
            attention = attention + self.tan(
                torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * alphas
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous().view(N,
                                                                                   self.num_subset_S * self.in_channels,
                                                                                   T, V)
        else:
            assert self.S_atten == 'free'
            attention = attention + self.tan(
                torch.einsum('nsctu,nsctv->nstuv', [q, k]) / (self.inter_channels)) * alphas
            attention = self.drop(attention)
            y = torch.einsum('nctu,nstuv->nsctv', [x, attention]).contiguous().view(N,
                                                                                   self.num_subset_S * self.in_channels,
                                                                                   T, V)

        y = self.out_nets(y)  # nctv
        y = self.relu(self.downs1(x) + y)
        y = self.ff_nets(y)
        s_out = self.relu(self.downs2(x) + y)


        # set_trace()
        # y_1 = self.out_nett_extend(y)
        # y_1 = self.relu(self.downt3(y) + y_1)
        # y = y_1
        if self.parallel:
            t_in = x
        else:
            t_in = s_out
        z = t_in
        if self.directed == True:
            forward_mask = self.backward_mask.transpose(-1, -2)
            backward_mask = self.backward_mask
            if self.T_atten == 'context_new':
                # print('T_context_new')
                q_k_in = self.in_nett(torch.mean(input=z,dim=-1,keepdim=True)).view(N, 4 * self.num_subset_T, self.inter_channels, T)
            else:
                q_k_in = self.in_nett(z).view(N, 4 * self.num_subset_T, self.inter_channels, T, V)
            if self.T_atten == 'context':
                # print('T_context')
                q_k_in = q_k_in.mean(-1)
            q_f, q_b, k_f, k_b = torch.chunk(q_k_in, 4, dim=1)

            if self.T_atten == 'context' or self.T_atten == 'context_new':
                attention_b = self.tan(torch.einsum('nsct,nscq->nstq', [q_b, k_b]) / (self.inter_channels)) * self.alphat_b
                attention_f = self.tan(torch.einsum('nsct,nscq->nstq', [q_f, k_f]) / (self.inter_channels)) * self.alphat_f
                attention_b = torch.einsum('nstq,tq->nstq', [attention_b, backward_mask])
                attention_f = torch.einsum('nstq,tq->nstq', [attention_f, forward_mask])
                attention_b = self.drop(attention_b)
                attention_f = self.drop(attention_f)
                z_f = torch.einsum('nctv,nstq->nscqv', [t_in, attention_f]).contiguous() \
                    .view(N, self.num_subset_T * self.in_channels_t, T, V)
                z_b = torch.einsum('nctv,nstq->nscqv', [t_in, attention_b]).contiguous() \
                    .view(N, self.num_subset_T * self.in_channels_t, T, V)
            elif self.T_atten == 'avg':
                attention_b = self.tan(torch.einsum('nsctv,nscqv->nstq', [q_b, k_b]) / (self.inter_channels * V)) * self.alphat_b
                attention_f = self.tan(torch.einsum('nsctv,nscqv->nstq', [q_f, k_f]) / (self.inter_channels * V)) * self.alphat_f
                attention_b = torch.einsum('nstq,tq->nstq', [attention_b, backward_mask])
                attention_f = torch.einsum('nstq,tq->nstq', [attention_f, forward_mask])
                attention_b = self.drop(attention_b)
                attention_f = self.drop(attention_f)
                z_f = torch.einsum('nctv,nstq->nscqv', [t_in, attention_f]).contiguous() \
                    .view(N, self.num_subset_T * self.in_channels_t, T, V)
                z_b = torch.einsum('nctv,nstq->nscqv', [t_in, attention_b]).contiguous() \
                    .view(N, self.num_subset_T * self.in_channels_t, T, V)
            else:
                assert self.T_atten == 'free'
                attention_b = self.tan(torch.einsum('nsctv,nscqv->nstqv', [q_b, k_b]) / (self.inter_channels)) * self.alphat_b
                attention_f = self.tan(torch.einsum('nsctv,nscqv->nstqv', [q_f, k_f]) / (self.inter_channels)) * self.alphat_f
                attention_b = torch.einsum('nstqv,tq->nstqv', [attention_b, backward_mask])
                attention_f = torch.einsum('nstqv,tq->nstqv', [attention_f, forward_mask])
                attention_b = self.drop(attention_b)
                attention_f = self.drop(attention_f)
                z_f = torch.einsum('nctv,nstqv->nscqv', [t_in, attention_f]).contiguous() \
                    .view(N, self.num_subset_T * self.in_channels_t, T, V)
                z_b = torch.einsum('nctv,nstqv->nscqv', [t_in, attention_b]).contiguous() \
                    .view(N, self.num_subset_T * self.in_channels_t, T, V)
            z = torch.cat([z_f, z_b], dim=-3)
        else:
            num_subset_temp = 2 * self.num_subset_T
            if self.T_atten == 'context_new':
                q_k_in = self.in_nett(torch.mean(input=z,dim=-1,keepdim=True)).view(N, 2 * num_subset_temp, self.inter_channels, T)
            else:
                q_k_in = self.in_nett(z).view(N, 2 * num_subset_temp, self.inter_channels, T, V)
            if self.T_atten == 'context':
                q_k_in = q_k_in.mean(-1)
            q, k = torch.chunk(q_k_in, 2, dim=1)
            if self.T_atten == 'context' or self.T_atten == 'context_new':
                attention_t = self.tan(torch.einsum('nsct,nscq->nstq', [q, k]) / (self.inter_channels)) * self.alphat
                attention_t = self.drop(attention_t)
                z = torch.einsum('nctv,nstq->nscqv', [y, attention_t]).contiguous().view(N, num_subset_temp * self.in_channels_t, T, V)
            elif self.T_atten == 'avg':
                attention_t = self.tan(torch.einsum('nsctv,nscqv->nstq', [q, k]) / (self.inter_channels * V)) * self.alphat
                attention_t = self.drop(attention_t)
                z = torch.einsum('nctv,nstq->nscqv', [y, attention_t]).contiguous().view(N, num_subset_temp * self.in_channels_t, T, V)
            else:
                assert self.T_atten == 'free'
                attention_t = self.tan(torch.einsum('nsctv,nscqv->nstqv', [q, k]) / (self.inter_channels)) * self.alphat
                attention_t = self.drop(attention_t)
                z = torch.einsum('nctv,nstqv->nscqv', [y, attention_t]).contiguous().view(N, num_subset_temp * self.in_channels_t, T, V)



        z = self.out_nett(z)  # nctv
        z = self.relu(self.downt1(t_in) + z)
        z = self.ff_nett(z)
        t_out = self.relu(self.downt2(t_in) + z)

        if self.parallel:
            z = s_out + t_out
        else:
            z = t_out
        # set_trace()
        if self.TCN==True:
            z_1 = self.out_nett_extend(z)
            z_1 = self.relu(self.downtTCN(z) + z_1)
            z = z_1
        return z


class GPTNet_option_ssl(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset_S=3, num_subset_T=3, dropout=0., config=None, num_person=2,
                 num_channel=3, glo_reg_s=True,
                 var=0.15, use_SSL=False, num_seg=3, directed=False, TCN=True, p_coord=0.2, p_coor_zero=0.8, p_coord_rand=0.1, p_sem=0.2, p_sem_zero=0.4, p_sem_rand=0.4,
                 attentiondrop=0, dropout2d=0, parallel=False, S_atten='free', T_atten='context', use_pet=True, use_pes=True, extra_data=True,
                 SSL_weight={'mask': 0.1, 'pred_S': 0.1, 'pred_T': 0.1, 'Contra': 0.1},
                 SSL_option={'mask': True, 'pred_S': True, 'pred_T': False, 'Contra': False}):
        super(GPTNet_option_ssl, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        self.var = var
        self.coord_channel = num_channel
        self.use_SSL = use_SSL
        self.SSL_weight = SSL_weight
        self.SSL_option = SSL_option
        self.num_seg = num_seg
        self.in_channels = in_channels
        self.num_person = num_person
        self.num_point = num_point
        self.num_frame = num_frame
        self.p_coord = p_coord
        self.p_coor_zero = p_coor_zero
        self.p_coord_rand = p_coord_rand
        self.p_sem = p_sem
        self.p_sem_zero = p_sem_zero
        self.p_sem_rand = p_sem_rand
        self.extra_data = extra_data
        self.use_pet = use_pet
        self.use_pes = use_pes
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        param = {
            'num_node': num_point,
            'num_subset_S': num_subset_S,
            'num_subset_T': num_subset_T,
            'glo_reg_s': glo_reg_s,
            'attentiondrop': attentiondrop,
            'directed': directed,
            'TCN': TCN,
            'parallel': parallel,
            'S_atten': S_atten,
            'T_atten': T_atten
        }
        self.graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(config):
            self.graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)

        self.fc = nn.Linear(self.out_channels, num_class)
        self.PES = nn.Conv2d(self.num_point, self.in_channels, 1, bias=False)
        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)
        self.init_pes()
        self.init_pet()
        self.init_SSL()



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def init_pes(self):
        pes = torch.eye(self.num_point)
        pes = pes[None,:,None,:]
        self.register_buffer('pes', pes)


    def init_pet(self):

        pet = torch.zeros(self.num_frame, self.in_channels).float()
        pet.require_grad = False
        position = torch.arange(0, self.num_frame).float().unsqueeze(1)
        div_term = (torch.arange(0, self.in_channels, 2).float() * -(math.log(10000.0) / self.in_channels)).exp()
        pet[:, 0::2] = torch.sin(position * div_term)
        pet[:, 1::2] = torch.cos(position * div_term)
        pet = pet.transpose(1,0)
        pet = pet[None,:,:,None]
        self.register_buffer('pet', pet)


    def init_SSL(self):
        # self.use_SSL = self.use_SSL and (np.array(list(self.SSL_option.values())) == True).any()
        if self.use_SSL == True:
            self.init_jigsaw()
            self.init_loss()
            self.SSL_mask = MaskedPrediction(hidden=self.out_channels, num_person=self.num_person, reconstruct=self.coord_channel)
            self.SSL_JigsawT = JigsawPrediction_T_labeled(hid_dim=self.out_channels, num_perm=self.num_jigsaw_T)
            self.SSL_JointP = Joint_Prediction(hid_dim=self.out_channels, num_joints=self.num_point)
            self.SSL_ReverseT = ReversePrediction(hidden = self.out_channels)
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
        reverse_T_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('reverse_T_loss_init', reverse_T_loss_init)





    def init_jigsaw(self):
        # initialize permutation for temporal dimensio
        temp_list_T = list(range(self.num_seg))
        self.permutations_T = list(itertools.permutations(temp_list_T))
        self.num_perm = len(self.permutations_T)
        idx = list(range(self.num_frame))
        cut_num = int(self.num_frame/self.num_seg)
        cut_idx = np.array([np.array(idx[i*cut_num:(i+1)*cut_num]) if i < self.num_seg-1 else np.array(idx[i*cut_num:]) for i in range(self.num_seg)])
        all_permute = np.arange(self.num_frame, 0, -1)[None,:]-1
        for i in range(self.num_perm):
            temp = np.concatenate(cut_idx[list(self.permutations_T[i])].tolist(),-1)[None,:]
            all_permute = np.concatenate((all_permute, temp), axis=0)
        self.num_jigsaw_T, length = all_permute.shape
        assert self.num_jigsaw_T == self.num_perm + 1
        all_permute = torch.from_numpy(all_permute)
        self.register_buffer('all_permute', all_permute)


    def Jigsaw_T_generate_labeled(self,x):
        # N, M, C, T, V = x.shape
        label_T = torch.randint_like(input=x[:,0,0,0,0], low=0, high=self.num_jigsaw_T, dtype=torch.long)
        permute_order = self.all_permute[label_T]
        permute_order = permute_order[:,None,None,:,None].expand(x.size())
        x_T = torch.gather(x,dim=-2,index=permute_order)
        return x_T, label_T




    def mask_coord(self,x):
        '''
        all samples share the same mask and noise
        Args:
            x: all input data

        Returns:

        '''
        N, M, C, T, V = x.shape
        noise = torch.randn_like(x)
        p_all = torch.rand_like(x[:,:,:1,:,:])
        threshold_coord = self.p_coord
        threshold_coord_zero = threshold_coord * self.p_coor_zero
        threshold_coord_rand = threshold_coord_zero + threshold_coord * self.p_coord_rand
        coord_all = (p_all <= threshold_coord).type_as(x)
        coord_zero = (p_all <= threshold_coord_zero).type_as(x)
        coord_rand = ((p_all>threshold_coord_zero) * (p_all <= threshold_coord_rand)).type_as(x)
        coord_keep = coord_all - coord_zero - coord_rand

        x_masked = x - x * coord_zero + coord_rand * noise

        return x_masked, coord_zero, coord_rand, coord_keep


    def mask_semantic(self, x, pes):
        N, M, C, T, V = x.shape

        N_s, C_s, T_s, V_s = pes.shape
        assert V==V_s and V== C_s and T_s==1 and N_s==1
        pes_sq = pes.squeeze()
        C_p, V_p = pes_sq.size()
        assert C_p==C_s and V_p==V_s
        p_all = torch.rand_like(x[:,0,:1,:1,:])
        threshold_sem = self.p_sem
        threshold_sem_zero = threshold_sem * self.p_sem_zero
        threshold_sem_rand = threshold_sem_zero + threshold_sem * self.p_sem_rand
        sem_all = (p_all <= threshold_sem).type_as(x)
        sem_zero = (p_all <= threshold_sem_zero).type_as(x)
        sem_rand = ((p_all>threshold_sem_zero) * (p_all<=threshold_sem_rand)).type_as(x)
        sem_keep = sem_all - sem_zero - sem_rand
        rand_index_S = torch.randint_like(input=x[:,0,0,0,:], low=0, high=V, dtype=torch.long)
        pes_rand = pes_sq[:,rand_index_S]
        pes_rand = pes_rand.transpose(1,0).unsqueeze(2)
        pes = pes.expand(N,C_s,1,V)
        pes_unkonwn = torch.rand_like(pes).softmax(dim=1)
        # pes_unkonwn = torch.ones_like(pes)/C_s
        pes_out = pes - pes*sem_all + pes_rand*sem_rand + pes*sem_keep + pes_unkonwn*sem_zero

        return pes_out, sem_zero, sem_rand, sem_keep


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
        reverse_loss = self.reverse_T_loss_init.expand(N)
        simloss = self.simloss_init.expand(N)
        pes = self.PES(self.pes).unsqueeze(1)
        pet = self.pet.unsqueeze(1)
        # set_trace()
        # if self.use_SSL == True and self.training == True:
        if self.use_SSL == True:
            # set_trace()
            x_origin = x.permute(0, 4, 1, 2, 3).contiguous()
            x_ssl = x_origin
            if self.SSL_option['mask'] == True:
                x_standard = x_ssl
                x_ssl, coord_zero, coord_rand, coord_keep = self.mask_coord(x_ssl)
            if self.SSL_option['pred_T'] == True:
                x_ssl, label_T = self.Jigsaw_T_generate_labeled(x_ssl)
            if self.SSL_option['pred_S'] == True:
                pes_ssl, sem_zero, sem_rand, sem_keep = self.mask_semantic(x_ssl.view(N, M, -1, T, V), self.pes)
                pes_ssl = self.PES(pes_ssl).unsqueeze(1)
            if self.extra_data == True:
                x = x_origin.view(N * M, C, T, V)
                x_ssl = x_ssl.view(N * M, C, T, V)
                x = self.input_map(x)
                x_ssl = self.input_map(x_ssl)
                x = (x.view(N,M,-1,T,V) + pet + pes).view(N*M,-1,T,V)
                x_ssl = (x_ssl.view(N,M,-1,T,V) + pet + pes_ssl).view(N*M,-1,T,V)
                '''Transformer blocks'''
                for i, m in enumerate(self.graph_layers):
                    x = m(x)
                    x_ssl = m(x_ssl)
            else:
                x_ssl = x_ssl.view(N * M, C, T, V)
                x_ssl = self.input_map(x_ssl)
                x_ssl = (x_ssl.view(N, M, -1, T, V) + pet + pes_ssl).view(N * M, -1, T, V)
                for i, m in enumerate(self.graph_layers):
                    x_ssl = m(x_ssl)

            # NM, C, T, V
            x_ssl = x_ssl.view(N, M, self.out_channels, T, V)
            '''predict the masked coordinates'''
            if self.SSL_option['mask'] == True:
                mask_loss = self.SSL_mask(x_standard=x_standard, x_masked=x_ssl, coord_zero=coord_zero, coord_rand=coord_rand, coord_keep=coord_keep)
            if self.SSL_option['pred_S'] == True:
                joint_loss = self.SSL_JointP(x=x_ssl,sem_zero=sem_zero, sem_rand=sem_rand, sem_keep=sem_keep)
            if self.SSL_option['pred_T'] == True:
                jigsaw_T_loss = self.SSL_JigsawT(x=x_ssl, labels_T=label_T)


            ''' downstream task(recognition)'''
            if self.extra_data == True:
                x = x.view(N, M, self.out_channels, -1)
            else:
                x = x_ssl.view(N, M, self.out_channels, -1)
            x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
            x = self.drop_out2d(x)
            x = x.mean(3).mean(1)

            x = self.drop_out(x)  # whole spatial of one channel


            pretext_loss = pretext_loss + self.SSL_weight['mask'] * mask_loss + self.SSL_weight['pred_S'] * joint_loss + \
                           self.SSL_weight['pred_T'] * jigsaw_T_loss + self.SSL_weight['reverse'] * reverse_loss + self.SSL_weight['Contra'] * simloss
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
            x = self.input_map(x)
            x = x + pet + pes
            # x = self.pet(self.pes(x))
            # x = self.pet(x)

            for i, m in enumerate(self.graph_layers):
                x = m(x)

            # NM, C, T, V

            x = x.view(N, M, self.out_channels, -1)
            x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
            x = self.drop_out2d(x)
            x = x.mean(3).mean(1)

            x = self.drop_out(x)  # whole spatial of one channel



        return self.fc(x), pretext_loss, mask_loss, jigsaw_T_loss, joint_loss, reverse_loss, simloss


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

