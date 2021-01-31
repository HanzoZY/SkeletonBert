import torch.nn as nn
import torch
# from .single import Attention
import torch.nn.functional as F
from ipdb import set_trace
import math

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, num_points=None, dropout=0.1, context=False):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.num_points = num_points
        self.context = context


        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

        # self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def attention_func(self, query, key, value, batch_check, mask=None, dropout=None,normal_A=None):
        N, Head, VoT, C = query.size()
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)


        if normal_A is not None:
            dim_check = normal_A.dim()
            for i in range(p_attn.dim()-dim_check):
                normal_A = normal_A.unsqueeze(0)
            p_attn = p_attn + normal_A

        if self.context == True:
            p_attn = p_attn.view(batch_check, -1, Head, VoT, VoT)
            ToV = p_attn.size(1)
            assert ToV * batch_check == N
            p_attn = p_attn.mean(1, keepdim=True)
            if dropout is not None:
                p_attn = dropout(p_attn)
            p_attn = p_attn.expand(-1, ToV, -1, -1, -1).contiguous().view(N, Head, VoT, VoT)
        else:
            if dropout is not None:
                p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, batch_check, mask=None,normal_A=None):
        batch_size = query.size(0)


        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # K,Q,V size: N(N*T or N*V), Head, V or T, C/Head
        # 2) Apply attention on all the projected vectors in batch.
        # x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout,normal_A=normal_A)
        x, attn = self.attention_func(query, key, value, batch_check, mask=mask, dropout=self.dropout, normal_A=normal_A)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
