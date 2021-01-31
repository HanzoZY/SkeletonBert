import torch
import torch.nn as nn
import math
import numpy as np
from ipdb import set_trace
from model.pretext_task import MaskedPrediction, JigsawPrediction_T, ContrastiveLearning_SimSiam, Joint_Prediction, ReversePrediction
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
    def __init__(self, in_channels, out_channels, inter_channels, num_subset_S=3, num_subset_T=2, num_node=25, num_frame=32,
                 kernel_size=7, stride=1, glo_reg_s=True, att_s=True, att_t=True, directed=True, TCN=True,affine_set=True,attentiondrop=0):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset_S = num_subset_S
        self.num_subset_T = num_subset_T
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.att_t = att_t
        self.directed = directed
        self.TCN = TCN

        backward_mask = torch.triu(torch.ones(num_frame, num_frame))
        self.register_buffer('backward_mask', backward_mask)

        pad = int((kernel_size - 1) / 2)

        # spatial operation
        atts = torch.zeros((1, num_node, num_node, self.num_subset_S))
        self.register_buffer('atts', atts)
        self.ff_nets = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=True),
            nn.LayerNorm(normalized_shape=[num_frame,out_channels], elementwise_affine=affine_set),# 是否应该只norm channel 而不是 channel+T or channel+S
        )
        self.in_nets = nn.Linear(in_channels, 2 * self.num_subset_S * inter_channels)
        self.alphas = nn.Parameter(torch.ones(1, 1, 1, self.num_subset_S), requires_grad=True)
        if glo_reg_s:
            self.attention0s = nn.Parameter(torch.ones(1, num_node, num_node, self.num_subset_S) / num_node,
                                            requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Linear(in_channels * self.num_subset_S, out_channels, bias=True),
            nn.LayerNorm(normalized_shape=[num_frame, out_channels],elementwise_affine=affine_set)
        )


        # temporal operation
        self.ff_nett = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(normalized_shape=[num_node, out_channels], elementwise_affine=affine_set)
        )
        self.in_nett = nn.Linear(out_channels, 4 * self.num_subset_T * inter_channels, bias=True)
        if self.directed == True:
            self.alphat_b = nn.Parameter(torch.ones(1, 1, 1, self.num_subset_T), requires_grad=True)
            self.alphat_f = nn.Parameter(torch.ones(1, 1, 1, self.num_subset_T), requires_grad=True)
        else:
            self.alphat = nn.Parameter(torch.ones(1, 1, 1, 2 * self.num_subset_T), requires_grad=True)
        # self.out_nett = nn.Sequential(
        #     nn.Conv2d(out_channels * self.num_subset_T * 2, out_channels, 1, bias=True),
        #     nn.BatchNorm2d(out_channels),
        # )
        self.out_nett = nn.Sequential(
            nn.Linear(out_channels * self.num_subset_T * 2, out_channels, bias=True),
            nn.LayerNorm(normalized_shape=[num_node, out_channels], elementwise_affine=affine_set)
        )
        if self.TCN==True:
            self.out_nett_extend = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(pad, 0), bias=True, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

        if in_channels != out_channels:
            self.downs1 = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=True),
                nn.LayerNorm(normalized_shape=[num_frame, out_channels],elementwise_affine=affine_set)
            )
            self.downs2 = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=True),
                nn.LayerNorm(normalized_shape=[num_frame, out_channels],elementwise_affine=affine_set)
            )

            self.downt1 = nn.Sequential(
                nn.Linear(out_channels, out_channels, bias=True),
                nn.LayerNorm(normalized_shape=[num_node, out_channels], elementwise_affine=affine_set),
            )
            self.downt2 = nn.Sequential(
                nn.Linear(out_channels, out_channels, bias=True),
                nn.LayerNorm(normalized_shape=[num_node, out_channels], elementwise_affine=affine_set),
            )
            if self.TCN==True:
                self.downt3 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, (1, 1), (1, 1), padding=(0, 0), bias=True),
                    nn.BatchNorm2d(out_channels),
                )
        else:
            self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt1 = lambda x: x
            self.downt2 = lambda x: x
            if self.TCN==True:
                self.downt3 = lambda x: x

        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.gelu = nn.GELU()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):

        # N, C, T, V = x.size()
        N, V, T, C = x.size()

        # spatial operation
        attention = self.atts
        y = x
        if self.att_s:
            q, k = torch.chunk(self.in_nets(y).view(N, V, T, self.inter_channels, 2 * self.num_subset_S), 2,
                               dim=-1)  # nvtc -> nvt num_subset c'
            attention = attention + self.tan(
                torch.einsum('nutcs,nvtcs->nuvs', [q, k]) / (self.inter_channels * T)) * self.alphas
        if self.glo_reg_s:
            attention = attention + self.attention0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        y = torch.einsum('nutc,nuvs->nvtcs', [x, attention]).contiguous().view(N, V, T, self.num_subset_S * self.in_channels)
        y = self.out_nets(y)  # nctv
        y = self.gelu(self.downs1(x) + y)
        y = self.ff_nets(y)
        y = self.gelu(self.downs2(x) + y)

        # temporal operation
        y = torch.einsum('nvtc->ntvc', [y])
        z = y
        if self.directed == True:
            forward_mask = self.backward_mask.transpose(-1, -2)
            backward_mask = self.backward_mask
            q_k_in = self.in_nett(z).view(N, T, V, self.inter_channels, 4 * self.num_subset_T)
            q_f, q_b, k_f, k_b = torch.chunk(q_k_in, 4, dim=-1)
            attention_b = self.tan(torch.einsum('ntvcs,nqvcs->ntqs', [q_b, k_b]) / (self.inter_channels * V)) * self.alphat_b
            attention_f = self.tan(torch.einsum('ntvcs,nqvcs->ntqs', [q_f, k_f]) / (self.inter_channels * V)) * self.alphat_f
            attention_b = torch.einsum('ntqs,tq->ntqs', [attention_b, backward_mask])
            attention_f = torch.einsum('ntqs,tq->ntqs', [attention_f, forward_mask])
            attention_b = self.drop(attention_b)
            attention_f = self.drop(attention_f)
            z_f = torch.einsum('ntvc,ntqs->nqvcs', [y, attention_f]).contiguous().view(N, T, V, self.num_subset_T * self.out_channels)
            z_b = torch.einsum('ntvc,ntqs->nqvcs', [y, attention_b]).contiguous().view(N, T, V, self.num_subset_T * self.out_channels)
            z = torch.cat([z_f, z_b], dim=-1)
        else:
            num_subset_temp = 2 * self.num_subset_T
            q_k_in = self.in_nett(z).view(N, T, V, self.inter_channels, 2 * num_subset_temp)
            q, k = torch.chunk(q_k_in, 2, dim=-1)
            attention_t = self.tan(torch.einsum('ntvcs,nqvcs->ntqs', [q, k]) / (self.inter_channels * V)) * self.alphat
            attention_t = self.drop(attention_t)
            z = torch.einsum('ntvc,ntqs->nqvcs', [y, attention_t]).contiguous().view(N, T, V, num_subset_temp * self.out_channels)

        z = self.out_nett(z)  # ntvc
        z = self.gelu(self.downt1(y) + z)
        z = self.ff_nett(z)
        z = self.gelu(self.downt2(y) + z)

        z = torch.einsum('ntvc->nvtc', [z])
        # set_trace()
        if self.TCN==True:
            z = torch.einsum('nvtc->nctv', [z])
            z_1 = self.out_nett_extend(z)
            z_1 = self.gelu(self.downt3(z) + z_1)
            z = torch.einsum('nctv->nvtc', [z_1])
        return z


class GPTNet_Lay(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=32, num_subset_S=3, num_subset_T=3, dropout=0., config=None, num_person=2,
                 num_channel=3, glo_reg_s=True, mask_divide=[0.8, 0.1, 0.1],
                 var=0.15, use_SSL=False, num_seg=3, directed=False, TCN=True, kernel_size=7, attentiondrop=0, dropout2d=0,
                 SSL_weight={'mask': 0.1, 'pred_S': 0.1, 'pred_T': 0.1, 'Contra': 0.1, 'reverse': 0.1},
                 SSL_option={'mask': True, 'pred_S': True, 'pred_T': False, 'Contra': True, 'reverse': False}):
        super(GPTNet_Lay, self).__init__()

        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        self.var = var
        self.use_SSL = use_SSL
        self.SSL_weight = SSL_weight
        self.SSL_option = SSL_option
        self.mask_divide = mask_divide
        self.num_seg = num_seg
        self.in_channels = in_channels
        self.num_person = num_person
        self.num_point = num_point
        self.num_frame = num_frame
        self.pet = PositionalEncoding(in_channels, num_point, num_frame, 'temporal')
        self.pes = PositionalEmbedding(in_channels, num_point, num_frame, 'spatial')
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
            'kernel_size': kernel_size
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
        self.init_SSL()


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
        self.SSL_ReverseT = ReversePrediction(hidden = self.out_channels)
        self.SSL_Contra = ContrastiveLearning_SimSiam(hid_dim=self.out_channels)


    def init_SSL(self):
        # self.use_SSL = self.use_SSL and (np.array(list(self.SSL_option.values())) == True).any()
        self.use_SSL = self.use_SSL
        self.init_jigsaw()
        self.init_loss()

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



    def Jigsaw_T_generate(self,x):
        N, C, T, V = x.shape
        idx = list(range(T))
        cut_num = int(T/self.num_seg)
        cut_idx = np.array([idx[i*cut_num:(i+1)*cut_num] if i < self.num_seg-1 else idx[i*cut_num:] for i in range(self.num_seg)])
        x_list = []
        num_perm = len(self.permutations_T)

        for i, idx_chose in enumerate(self.permutations_T):
            idx_i_permute = cut_idx[list(idx_chose)].tolist()
            idx_i = [j for k in idx_i_permute for j in k]
            x_list.append(x[:,:,idx_i,:])

        x_T = torch.stack(x_list)
        assert num_perm == x_T.size(0)
        x_T = x_T.view(num_perm, N, C, T, V)

        return x_T

    def reverse_T_generate(self, x):
        N, C, T, V = x.shape
        assert T == self.num_frame
        x_reverse = torch.flip(x, dims=[-2])
        assert x[0,0,0,0] == x_reverse[0,0,-1,0]
        return x_reverse



    def random_mask_all(self,x):
        '''
        all samples share the same mask and noise
        Args:
            x: all input data

        Returns:

        '''
        N, C, T, V = x.shape
        noise = torch.FloatTensor(*x.size()[2:]).uniform_(-self.var, self.var).unsqueeze(0)
        masks = torch.tensor(data=get_mask_array(size_in=V*T, divide_rate=self.mask_divide),dtype=torch.float)

        # noise = noise.to(x.get_device())
        # masks = masks.to(x.get_device())

        x_masked = x
        num_mask = masks.size(0)
        # set_trace()
        assert num_mask==len(self.mask_divide)
        for i, mask in enumerate(masks.chunk(num_mask, dim=0)):
            if i==0:
                continue
            if i==1:
                mask = 1-mask.view(T,V).unsqueeze(0).unsqueeze(0)
                x_masked = x_masked * mask
            if i==2:
                mask_noise = mask.view(T,V).unsqueeze(0).unsqueeze(0) * noise
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
        reverse_loss = self.reverse_T_loss_init.expand(N)
        simloss = self.simloss_init.expand(N)




        # set_trace()
        # if self.use_SSL == True and self.training == True:
        if self.use_SSL == True:
            # set_trace()
            x_origin = x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V)
            if self.SSL_option['mask'] == True:
                x_masked = self.random_mask_all(x=x_origin).detach()
                x_masked = x_masked.view(N * M, C, T, V)

            x_temp = self.input_map(x_origin)
            x = self.pet(self.pes(x_temp))
            if self.SSL_option['mask'] == True:
                x_masked = self.input_map(x_masked)
                x_masked = self.pet(self.pes(x_masked))
            if self.SSL_option['pred_T'] == True:
                x_jigsaw_T = self.Jigsaw_T_generate(x=x_temp.view(N*M, self.in_channels, T, V))
                num_perm = x_jigsaw_T.size(0)
                assert num_perm == len(self.permutations_T)
                x_jigsaw_T = x_jigsaw_T.view(num_perm * N * M, self.in_channels, T, V)
                x_jigsaw_T = self.pet(self.pes(x_jigsaw_T))
            if self.SSL_option['pred_S'] == True:
                x_predict_S = self.pet(x_temp)
            if self.SSL_option['reverse'] == True:
                x_reverse = self.reverse_T_generate(x_temp.view(N*M, self.in_channels, T, V))
                x_reverse = x_reverse.view(N * M, self.in_channels, T, V)
                x_reverse = self.pet(self.pes(x_reverse))

            '''Transformer blocks'''

            for i, m in enumerate(self.graph_layers):
                x = m(x)
                if self.SSL_option['mask'] == True:
                    x_masked = m(x_masked)
                if self.SSL_option['pred_T'] == True:
                    x_jigsaw_T = m(x_jigsaw_T)
                if self.SSL_option['pred_S'] == True:
                    x_predict_S = m(x_predict_S)
                if self.SSL_option['reverse'] == True:
                    x_reverse = m(x_reverse)
            # NM, C, T, V

            '''predict the masked coordinates'''
            x_all = []
            if self.SSL_option['mask'] == True:
                mask_loss = self.SSL_mask(x_origin=x_origin, x_masked=x_masked)
                x_masked = x_masked.view(N, M, self.out_channels, T, V)
                x_masked = x_masked.mean(-1).mean(-1).mean(1).view(N,self.out_channels).unsqueeze(0)
                x_all.append(x_masked)


            '''predict the Jigsaw T'''
            if self.SSL_option['pred_T'] == True:
                x_jigsaw_T = x_jigsaw_T.view(num_perm, N, M, self.out_channels, T, V)
                x_jigsaw_T = x_jigsaw_T.mean(-1).mean(-1).mean(2).view(num_perm, N, self.out_channels)
                jigsaw_T_loss = self.SSL_JigsawT(x=x_jigsaw_T)
                x_all.append(x_jigsaw_T)



            '''predict the joint type'''
            if self.SSL_option['pred_S'] == True:
                x_predict_S = x_predict_S.view(N, M, self.out_channels, T, V)
                x_predict_S = x_predict_S.mean(-2).mean(1).view(N, self.out_channels, V)
                joint_loss = self.SSL_JointP(x=x_predict_S)
                x_predict_S = x_predict_S.mean(-1).view(N, self.out_channels).unsqueeze(0)
                x_all.append(x_predict_S)

            '''predict the reverse sample'''
            if self.SSL_option['reverse'] == True:
                x_reverse = x_reverse.view(N, M, self.out_channels, T*V)
                x_reverse = x_reverse.mean(-1).mean(1).view(N, self.out_channels)
                reverse_loss = self.SSL_ReverseT(x=x.view(N, M, self.out_channels, T*V).mean(-1).mean(1).view(N, self.out_channels), x_reverse=x_reverse)
                x_all.append(x_reverse.unsqueeze(0))



            ''' ContrastiveLearning '''
            if self.SSL_option['Contra'] == True:
                x4SSL = x.view(N, M, self.out_channels, T*V).mean(-1).mean(1).view(N, self.out_channels).unsqueeze(0)
                x_all.append(x4SSL)
                x_all = torch.cat(x_all,dim=0)
                simloss = self.SSL_Contra(x_all)



            ''' downstream task(recognition)'''
            x = x.view(N, M, self.out_channels, -1)
            x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)  # whole channels of one spatial
            x = self.drop_out2d(x)
            x = x.mean(3).mean(1)

            x = self.drop_out(x)  # whole spatial of one channel

            pretext_loss = pretext_loss + self.SSL_weight['mask'] * mask_loss + self.SSL_weight['pred_S'] * joint_loss + \
                           self.SSL_weight['pred_T'] * jigsaw_T_loss + self.SSL_weight['reverse'] * reverse_loss + self.SSL_weight['Contra'] * simloss
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
            x = self.input_map(x)
            x = self.pet(self.pes(x))
            # x = self.pet(x)

            x = torch.einsum('nctv->nvtc',[x])
            for i, m in enumerate(self.graph_layers):
                x = m(x)

            # NM, C, T, V
            x = torch.einsum('nvtc->nctv', [x])
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

