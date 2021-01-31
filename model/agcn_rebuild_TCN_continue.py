import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from ipdb import set_trace

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        # kenel size: the convlution only op on temporal dimention
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.out_channels = out_channels

    def forward(self, x):
        # N, C, T, V = x.size()
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, C, T, V)
        x = self.bn(self.conv(x))
        return x.permute(0, 2, 3, 1).contiguous().view(N, -1, V, self.out_channels)


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_channels = out_channels
        self.PA = nn.Parameter(torch.zeros(num_subset,num_nodes,num_nodes))
        nn.init.constant_(self.PA, 1e-6) # PA is the trainable mask?
        # self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x, A):
        # N, C, T, V = x.size()
        # set_trace()
        N, T, V, C = x.size()
        # A = self.A.cuda(x.get_device())
        # A = self.A.to(x.get_device())
        A = A + self.PA
        x = x.permute(0, 3, 1, 2).contiguous().view(N, C, T, V)
        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y).permute(0, 2, 3, 1).contiguous().view(N, T, V, self.out_channels)


class unit_gcn_confirm(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, coff_embedding=4, num_subset=3):
        super(unit_gcn_confirm, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_channels = out_channels
        self.PA = nn.Parameter(torch.zeros(num_subset, num_nodes, num_nodes))
        nn.init.constant_(self.PA, 1e-6)  # PA is the trainable mask?
        self.num_subset = num_subset
        self.weight_A = nn.Parameter(torch.tensor([0.25,0.25,0.25,0.25],dtype=torch.float),requires_grad=True)

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_a_1 = nn.ModuleList()
        self.conv_b_1 = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            # self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            # self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            # self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            self.conv_a.append(nn.Linear(in_channels, inter_channels))
            self.conv_b.append(nn.Linear(in_channels, inter_channels))

            self.conv_a_1.append(nn.Linear(in_channels, inter_channels))
            self.conv_b_1.append(nn.Linear(in_channels, inter_channels))

            self.conv_d.append(nn.Linear(in_features=in_channels, out_features=out_channels))



        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-1)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        # for i in range(self.num_subset):
        #     conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x, A):
        # N, C, T, V = x.size()
        # set_trace()
        N, T, V, C = x.size()
        x_temp = x
        PA = torch.softmax(self.PA,dim=-1)
        # A = self.A.cuda(x.get_device())
        # A = self.A.to(x.get_device())
        A = self.weight_A[0]*A + self.weight_A[1]*PA
        x = x.permute(0, 3, 1, 2).contiguous().view(N, C, T, V)
        y = None
        for i in range(self.num_subset):
            # A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            # A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            q1 = self.conv_a[i](x_temp).permute(0, 2, 3, 1).contiguous().view(N, V, self.inter_c * T)
            k1 = self.conv_b[i](x_temp).permute(0, 3, 1, 2).contiguous().view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(q1, k1) / math.sqrt(q1.size(-1)))  # N V V
            # A1 = self.weight_A[2]*A1 + A[i]

            # A1 = A1.transpose(-1, -2)
            # A2 = x_temp

            # A2 = x.view(N, C * T, V)

            A1 = A1.unsqueeze(1)
            A1 = A1.expand(-1, T, -1, -1)

            q2 = self.conv_a[i](x_temp)
            k2 = self.conv_b[i](x_temp).permute(0, 1, 3, 2).contiguous().view(N, T, self.inter_c, V)
            A2 = self.soft(torch.matmul(q2, k2) / math.sqrt(q2.size(-1)))
            A_in = A[i]+self.weight_A[2]*A1+self.weight_A[3]*A2
            # z = torch.matmul(A1, A2)

            # z = torch.matmul(A2, A1).view(N, C, T, V).permute(0,2,3,1).contiguous().view(N,T,V,C)
            z = torch.matmul(A_in, x_temp)
            z = self.conv_d[i](z)
            z = z.permute(0, 3, 1, 2).contiguous().view(N, -1, T, V)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y).permute(0, 2, 3, 1).contiguous().view(N, T, V, self.out_channels)



class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn = unit_gcn_confirm(in_channels, out_channels, num_nodes)
        self.tcn = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A):
        x = self.tcn(self.gcn(x=x, A=A)) + self.residual(x)
        return self.relu(x)


class AGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, in_channels=3, num_super_nodes=6, add_before_bn=True, add_idx=True, num_subset=3, out_channel=256):
        super(AGCN, self).__init__()
        self.sub_set = num_subset
        self.PA = nn.Parameter(torch.zeros(self.sub_set, num_point, num_point))
        pretext_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('pretext_loss_init', pretext_loss_init)
        nn.init.constant_(self.PA, 1e-6)
        self.add_before_bn = add_before_bn
        self.add_idx = add_idx
        self.S_0 = nn.Parameter(torch.Tensor(num_point, num_super_nodes))
        self.S_1 = nn.Parameter(torch.Tensor(num_super_nodes, 1))
        nn.init.xavier_uniform_(self.S_0)
        nn.init.xavier_uniform_(self.S_1)
        label_ebd = torch.eye(num_point, requires_grad=False, dtype=torch.float)
        self.register_buffer('label_ebd', label_ebd)

        if self.add_before_bn and self.add_idx:
            self.data_bn = nn.BatchNorm1d(num_person * (in_channels+num_point) * num_point)
        else:
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        if self.add_idx:
            self.l1 = TCN_GCN_unit(in_channels + num_point, 64, num_nodes=num_point, residual=False)
        else:
            self.l1 = TCN_GCN_unit(in_channels, 64, num_nodes=num_point, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, num_nodes=num_point)
        self.l3 = TCN_GCN_unit(64, 64, num_nodes=num_point)
        self.l4 = TCN_GCN_unit(64, 64, num_nodes=num_point)
        self.l5 = TCN_GCN_unit(64, 128, num_nodes=num_point, stride=2)

        self.l6_0 = TCN_GCN_unit(128, 128, num_nodes=num_point)
        self.l7_0 = TCN_GCN_unit(128, 128, num_nodes=num_point)
        self.l8_0 = TCN_GCN_unit(128, 256, num_nodes=num_point, stride=2)

        self.l6_1 = TCN_GCN_unit(128, 128, num_nodes=num_super_nodes)
        self.l7_1 = TCN_GCN_unit(128, 128, num_nodes=num_super_nodes)
        self.l8_1 = TCN_GCN_unit(128, 256, num_nodes=num_super_nodes, stride=2)

        self.l9 = TCN_GCN_unit(256, 256, num_nodes=1)
        self.l10 = TCN_GCN_unit(256, 256, num_nodes=1)

        self.l9_0 = TCN_GCN_unit(256, 256, num_nodes=num_point)
        self.l10_0 = TCN_GCN_unit(256, 256, num_nodes=num_point)

        self.l9_1 = TCN_GCN_unit(256, 256, num_nodes=num_super_nodes)
        self.l10_1 = TCN_GCN_unit(256, 256, num_nodes=num_super_nodes)

        self.l9_2 = TCN_GCN_unit(256, 256, num_nodes=1)
        self.l10_2 = TCN_GCN_unit(256, 256, num_nodes=1)



        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def add_and_bn(self,x):
        N, C, T, V, M = x.size()
        if self.add_idx:
            if self.add_before_bn == True:
                # set_trace()
                x = x.permute(0, 4, 2, 3, 1).contiguous().view(N * M, T, V, C)
                # x = p_lb_ebd(x=x, I=self.label_ebd.cuda(x.get_device()))
                x = p_lb_ebd(x=x, I=self.label_ebd)
                C_new = x.size(-1)
                x = x.view(N, M, T, V, C_new).permute(0, 1, 3, 4, 2).contiguous().view(N, M*V*C_new, T)
                x = self.data_bn(x)
                x = x.view(N, M, V, C_new, T).permute(0, 1, 4, 2, 3).contiguous().view(N * M, T, V, C_new)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
                x = self.data_bn(x)
                x = x.view(N, M, V, C, T).permute(0, 1, 4, 2, 3).contiguous().view(N * M, T, V, C)
                # x = p_lb_ebd(x=x, I=self.label_ebd.cuda(x.get_device()))
                x = p_lb_ebd(x=x, I=self.label_ebd)
        else:

            x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T).permute(0, 1, 4, 2, 3).contiguous().view(N * M, T, V, C)
        return x


    def forward(self, x):

        S_0 = torch.softmax(self.S_0, dim=0)
        S_1 = torch.softmax(self.S_1, dim=0)
        pretext_loss = self.pretext_loss_init
        N, C, T, V, M = x.size()
        x = self.add_and_bn(x)
        K,T,V,C = x.size()
        # A = self.A.cuda(x.get_device())
        A = torch.softmax(self.PA,dim=-1)
        pretext_loss_init = torch.zeros(1, requires_grad=False)
        self.register_buffer('pretext_loss_init', pretext_loss_init)
        x = self.l1(x, A)
        x = self.l2(x, A)
        x = self.l3(x, A)
        x = self.l4(x, A)
        x = self.l5(x, A)
        # set_trace()
        x_1, A_1 = p_coarsen(x=x, s=S_0), p_coarsen_A(A=A, s=S_0)
        x,x_1 = self.l6_0(x, A),self.l6_1(x_1,A_1)
        x,x_1 = self.l7_0(x, A),self.l7_1(x_1,A_1)
        x,x_1 = self.l8_0(x, A),self.l8_1(x_1,A_1)
        x_2, A_2 = p_coarsen(x=x_1, s=S_1), p_coarsen_A(A=A_1, s=S_1)
        x, x_1, x_2 = self.l9_0(x, A),self.l9_1(x_1,A_1),self.l9_2(x_2,A_2)
        x, x_1, x_2 = self.l10_0(x, A),self.l10_1(x_1,A_1),self.l10_2(x_2,A_2)


        # N*M,C,T,V
        # c_new = x.size(1)
        # x = x.view(N, M, c_new, -1)
        # x = x.mean(3).mean(1)

        # N*M, T, V, C
        # set_trace()
        x = torch.cat((x,x_1,x_2),dim=-2)
        K, T, V, C_new = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N*M, C_new, T, V)
        x = x.view(N, M, C_new, -1)
        x = x.mean(3).mean(1)


        return self.fc(x), pretext_loss.unsqueeze(0).expand(N, -1)


def p_coarsen(x,s):
    N, T, V, C = x.size()
    s = s.unsqueeze(0) if s.dim() == 2 else s
    B, V_1, V_2 = s.size()
    assert V_1 == V
    x_temp = x.view(N*T, V, C)
    x_temp = torch.matmul(s.transpose(-1, -2), x_temp).view(N, T, V_2, C)
    return x_temp



def p_coarsen_A(A, s):
    s = s.unsqueeze(0) if s.dim() == 2 else s
    A_out = torch.matmul(torch.matmul(s.transpose(1, 2), A), s)
    return A_out


def p_lb_ebd(x, I):
    N, T, V ,C = x.size()
    x_temp = x
    x_temp = torch.cat((x_temp, I.repeat(N,T,1,1)),dim=-1)
    C_new = x_temp.size(-1)
    assert C_new == V+C
    return x_temp

