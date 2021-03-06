import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from ipdb import set_trace
'''
all the SSL task should merge all subjects together except the coordinates prediction task!!!


'''





def save_grad(name):
    def hook(grad):
        tag = name+' grad is'
        print(tag, grad)
    return hook
class JigsawPrediction_T(nn.Module):
    """
    jigsaw puzzle recognition task
    """

    def __init__(self, hid_dim, num_perm):
        """
        :param hidden: BERT model output size
        """
        super(JigsawPrediction_T, self).__init__()
        self.num_perm = num_perm
        Jigsaw_label = torch.tensor(list(range(self.num_perm)),dtype=torch.long)
        Jigsaw_label.requires_grad = False
        self.register_buffer('Jigsaw_label', Jigsaw_label)
        self.hid_dim = hid_dim
        self.MLP = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.num_perm))
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    # def forward(self, x):
    #     """
    #     return the mean loss of each jigsaw puzzled sample
    #     """
    #     N_all, T, V, C = x.size()
    #     assert N_all%self.num_perm == 0
    #     N = int(N_all/self.num_perm)
    #     # assert C_label == self.num_perm
    #     x = x.mean(1).mean(1)
    #     x = self.linear(x)
    #     # label = self.label.unsqueeze(-1).expand(-1,N).reshape(self.num_perm*N).to(x.get_device())
    #     label = self.Jigsaw_label.unsqueeze(-1).expand(-1,N).reshape(self.num_perm*N)
    #     loss_Jigsaw_T = self.loss_func(input=x, target=label)
    #     return loss_Jigsaw_T

    def forward(self, x):
        """

        Args:
            x: num_perm, batchsize, channel

        Returns:
            loss of each sample

        """
        num_perm, N, C  = x.size()
        assert num_perm == self.num_perm
        x = x.view(num_perm*N, C)
        x = self.MLP(x)
        x = x.view(num_perm*N, self.num_perm)
        label = self.Jigsaw_label.unsqueeze(-1).expand(-1,N).reshape(self.num_perm*N)
        loss_Jigsaw_T = self.loss_func(input=x, target=label)
        loss_Jigsaw_T = loss_Jigsaw_T.view(num_perm, N)
        loss_Jigsaw_T = loss_Jigsaw_T.mean(0)
        return loss_Jigsaw_T

class Joint_Prediction(nn.Module):
    """
    jigsaw puzzle recognition task
    """

    def __init__(self, hid_dim, num_joints):
        """
        :param hidden: BERT model output size
        """
        super(Joint_Prediction, self).__init__()
        self.num_joints = num_joints
        self.hid_dim = hid_dim
        self.MLP = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.num_joints))
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        # label = torch.eye(num_joints)
        # self.label = Variable(torch.tensor(list(range(self.num_joints)),dtype=torch.long), requires_grad=False)
        Joints_label = torch.tensor(list(range(self.num_joints)),dtype=torch.long)
        Joints_label.requires_grad = False
        self.register_buffer('Joints_label', Joints_label)


    # def forward(self, x):
    #     """
    #     return the mean loss of each joint puzzled joint
    #     """
    #     N, T, V, C = x.size()
    #     assert V == self.num_joints
    #     x = x.mean(1)
    #     x = x.view(N*V, C)
    #     label = self.Joints_label.unsqueeze(0).expand(N,-1).reshape(N*V)
    #     x = self.linear(x)
    #     Joint_loss = self.loss_func(input=x, target=label)
    #     return Joint_loss

    def forward(self, x):
        """
        return the mean loss of each joint puzzled joint
        """
        N, C, V = x.size()
        assert V == self.num_joints
        x = x.permute(0, 2, 1).contiguous().view(N*V,C)
        x = self.MLP(x)
        label = self.Joints_label.unsqueeze(0).expand(N,-1).reshape(N*V)
        Joint_loss = self.loss_func(input=x, target=label).view(N,V)
        Joint_loss = Joint_loss.mean(1)
        return Joint_loss





class MaskedPrediction(nn.Module):
    """
    predicting origin coordinates from masked input sequence
    """

    def __init__(self, hidden, num_person, reconstruct=3):
        """
        :param hidden: output size of BERT model
        :reconstruct: the original channel of coordinates
        """
        super(MaskedPrediction, self).__init__()
        self.num_person = num_person
        self.reconstruct = reconstruct
        self.hid_dim = hidden
        self.MLP = nn.Sequential(nn.Conv2d(self.hid_dim, self.hid_dim, 1, 1, padding=0, bias=True),
                                 nn.BatchNorm2d(num_features=self.hid_dim),
                                 nn.LeakyReLU(), nn.Conv2d(self.hid_dim, reconstruct, 1, 1, padding=0, bias=True))
        # self.loss_func = torch.nn.MSELoss(reduction='none')
        self.loss_func = torch.nn.L1Loss(reduction='none')


    def forward(self, x_origin, x_masked):

        """

        Args:
            x_origin: the original coordinates of joints
            x_masked: the embedding of all joints including normal, masked, and noised

        Returns:
            the loss of maskprediction of each sample N \times 1
        """
        N_0,C,T,V = x_origin.size()
        # set_trace()
        assert C==self.reconstruct
        x_origin = x_origin.detach()
        x_masked = self.MLP(x_masked)
        loss_all = self.loss_func(input=x_masked, target=x_origin)
        loss_all = loss_all.view(-1, self.num_person*C*T*V)
        loss_all = loss_all.mean(-1)
        return loss_all




class ReversePrediction(nn.Module):
    """
    predicting the direction of sample on temporal dimension
    2-class classification problem
    """

    def __init__(self, hidden):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super(ReversePrediction, self).__init__()
        self.num_class = 2
        reverse_label = torch.tensor(list(range(self.num_class)), dtype=torch.long)
        self.register_buffer('reverse_label', reverse_label)

        self.hid_dim = hidden
        self.MLP = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.num_class))
        # self.loss_func = torch.nn.MSELoss(reduction='none')
        # self.loss_func = torch.nn.L1Loss(reduction='none')
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')


    def forward(self, x, x_reverse):

        """

        Args:
            x_origin: the original coordinates of joints
            x_masked: the embedding of all joints including normal, masked, and noised

        Returns:
            the loss of maskprediction of each sample N \times 1
        """
        N_0, C_0 = x.size()
        N_1, C_1 = x_reverse.size()
        assert N_0==N_1 and C_0==C_1
        # set_trace()
        # x_all = torch.stack([x,x_reverse],dim=0)
        x_all = torch.cat([x, x_reverse], dim=0)
        label = self.reverse_label.unsqueeze(-1).expand(-1,N_0).reshape(self.num_class*N_0)
        x_all = x_all.view(self.num_class*N_0, C_0)
        x_all = self.MLP(x_all)
        loss_reverse = self.loss_func(input=x_all, target=label)
        loss_reverse = loss_reverse.view(self.num_class, N_0)
        loss_reverse = loss_reverse.mean(0)

        return loss_reverse




class ContrastiveLearning_SimCLR(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, num_perm):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super(ContrastiveLearning_SimCLR, self).__init__()
        self.sim = torch.nn.CosineSimilarity(dim=-1)
        self.num_perm = num_perm
        self.log_sm = torch.nn.LogSoftmax(dim=0)


    def forward(self, x, x_masked, x_jigsaw_T, x_predict_S):
        """
        return the mean loss of each generated sample

        """

        N,T,V,C = x.size()

        N_masked, T_masked, V_masked, C_masked = x_masked.size()

        N_jigsaw_T, T_jigsaw_T, V_jigsaw_T, C_jigsaw_T = x_jigsaw_T.size()

        N_predict_S, T_predict_S, V_predict_S, C_predict_S = x_predict_S.size()

        assert N_masked == N and N_predict_S == N
        assert N * self.num_perm == N_jigsaw_T

        x_jigsaw_T = x_jigsaw_T.view(self.num_perm, N, T_jigsaw_T, V_jigsaw_T, C_jigsaw_T)
        # x_jigsaw_T = x_jigsaw_T.permute(1,0,2,3,4).contiguous().view(N, self.num_perm, T_jigsaw_T, V_jigsaw_T, C_jigsaw_T)
        x = x.unsqueeze(0)
        x_masked = x_masked.unsqueeze(0)
        x_predict_S = x_predict_S.unsqueeze(0)
        x_all = torch.cat((x,x_masked,x_jigsaw_T,x_predict_S),dim=0)
        x_all = x_all.mean(2).mean(2)
        x_all = x_all.permute(1, 0, 2).view(N, -1, C).contiguous()
        x_bar = x_all.mean(1, keepdim=True)
        N, K, C = x_all.size()
        x_flatten = x_all.view(N*K,C)

        # sim_matrix: N \times (N \times K)
        sim_matrix = self.sim(x_bar, x_flatten)
        N_sim, F_sim = sim_matrix.size()
        assert N_sim == N and F_sim == N * K
        sim_matrix = sim_matrix.view(N,N,K)
        sim_score_matrix = self.log_sm(sim_matrix)
        # hit_map = torch.eye(N).to(device=x.get_device()).unsqueeze(-1)
        hit_map = torch.eye(N).unsqueeze(-1)
        sim_score_hit = hit_map * sim_score_matrix
        contrastive_loss = sim_score_hit.sum() / F_sim

        return -contrastive_loss



class ContrastiveLearning_SimSiam(nn.Module):
    """

    """

    def __init__(self, hid_dim):
        """
        :param hid_dim: output size of BERT model
        """
        super(ContrastiveLearning_SimSiam, self).__init__()
        self.hid_dim = hid_dim
        self.projection = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim),
                                        nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.hid_dim), nn.LeakyReLU(),
                                        nn.BatchNorm1d(num_features=self.hid_dim))
        self.prediction = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.BatchNorm1d(num_features=self.hid_dim),
                                        nn.LeakyReLU(), nn.Linear(self.hid_dim, self.hid_dim))




    def SimSiamLoss(self,p,z):

        K_p, N_p, C_p = p.size()
        K_z, N_z, C_z = z.size()
        z = z.detach()
        p = F.normalize(p, dim=-1)  # l2-normalize
        z = F.normalize(z, dim=-1)  # l2-normalize
        similarity_map = torch.einsum('pnc,znc->pznc', [p, z])
        simloss = -similarity_map.sum(-1).mean(0).mean(0)
        return simloss




    def forward(self, x):
        """
        return the mean loss of each generated sample

        """
        K, N, C = x.size()
        x = x.view(K*N,C)
        z = self.projection(x)
        p = self.prediction(z)
        z = z.view(K, N, C)
        p = p.view(K, N, C)
        simloss = self.SimSiamLoss(z=z, p=p)
        return simloss