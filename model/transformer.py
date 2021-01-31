import torch.nn as nn
import torch
from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward, gelu
from ipdb import set_trace

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden_in, hidden_out, attn_heads, num_joints, num_frames, feed_forward_hidden, dropout, attentiondrop, context=False, unique_layer_S_A=True, unique_layer_T_A=False):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.context = context
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.unique_layer_S_A = unique_layer_S_A
        self.unique_layer_T_A = unique_layer_T_A
        if self.unique_layer_S_A == True:
            self.unique_S_attention_map = nn.Parameter(torch.ones(1, attn_heads, num_joints, num_joints) / num_joints,
                                                requires_grad=True)
        else:
            self.unique_S_attention_map = None

        if self.unique_layer_T_A == True:
            self.unique_T_attention_map = nn.Parameter(torch.zeros(1, attn_heads, num_frames, num_frames) + torch.eye(num_frames),
                                                requires_grad=True)
        else:
            self.unique_T_attention_map = None


        # self-attention model on spatial dimension
        self.attention_S = MultiHeadedAttention(h=attn_heads, d_model=hidden_in, num_points=num_joints, context=self.context, dropout=attentiondrop)
        self.feed_forward_S = PositionwiseFeedForward(d_in=hidden_in, d_ff=feed_forward_hidden, d_out=hidden_in,dropout=dropout)
        self.input_sublayer_S = SublayerConnection(size_in=hidden_in, size_out=hidden_in, dropout=dropout, for_input_sublayer=True)
        self.output_sublayer_S = SublayerConnection(size_in=hidden_in, size_out=hidden_in, dropout=dropout)
        # to do:  all frames share one attention map
        # self.attention_S_context

        # self-attetnion model on temporal dimension
        self.attention_T = MultiHeadedAttention(h=attn_heads, d_model=hidden_in, num_points=num_frames, context=self.context, dropout=attentiondrop)
        self.feed_forward_T = PositionwiseFeedForward(d_in=hidden_in, d_ff=feed_forward_hidden, d_out=hidden_out, dropout=dropout)
        self.input_sublayer_T = SublayerConnection(size_in=hidden_in, size_out=hidden_in, dropout=dropout, for_input_sublayer=True)
        self.output_sublayer_T = SublayerConnection(size_in=hidden_in, size_out=hidden_out, dropout=dropout)

        if hidden_out == hidden_in:
            self.down = lambda x: x
        else:
            self.down = nn.Linear(hidden_in, hidden_out)

        # to do: all joints share one attention map
        # self.attention_T_context

        self.dropout = nn.Dropout(p=dropout)
        self.activation = gelu.GELU()

    def forward(self, x, R_A=None, mask=None):
        # self-attetnion on spatial dimension
        N, T, V, C = x.size()
        x = x.view(N*T, V, C)
        if R_A is None:
            if self.unique_layer_S_A == True:
                R_A = self.unique_S_attention_map
        else:
            if self.unique_layer_S_A == True:
                R_A = self.unique_S_attention_map + R_A


        x = self.input_sublayer_S(x=x, sublayer=lambda _x, A_G: self.attention_S.forward(_x, _x, _x, batch_check=N, mask=mask, normal_A=A_G), A=R_A)
        x = self.output_sublayer_S(x, self.feed_forward_S)

        # self-attention on temporal dimension
        C = x.size(-1)
        x = x.view(N, T, V, C)
        x = x.permute(0,2,1,3).contiguous().view(N*V, T, C)
        x = self.input_sublayer_T(x=x, sublayer=lambda _x, A_G: self.attention_T.forward(_x, _x, _x, batch_check=N, mask=mask, normal_A=A_G), A=self.unique_T_attention_map)
        x = self.output_sublayer_T(x, self.feed_forward_T)
        C = x.size(-1)
        x = x.view(N, V, T, C).permute(0,2,1,3).contiguous().view(N, T, V, C)

        return self.dropout(x)



# TransformerBlock_context unfinished
# class TransformerBlock_context(nn.Module):
#     """
#     Bidirectional Encoder = Transformer (self-attention)
#     Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
#     """
#
#     def __init__(self, hidden_in, hidden_out, attn_heads, feed_forward_hidden, dropout):
#         """
#         :param hidden: hidden size of transformer
#         :param attn_heads: head sizes of multi-head attention
#         :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
#         :param dropout: dropout rate
#         """
#
#         super().__init__()
#
#         # self-attention model on spatial dimension
#         self.attention_S = MultiHeadedAttention(h=attn_heads, d_model=hidden_in)
#         self.feed_forward_S = PositionwiseFeedForward(d_in=hidden_in, d_ff=feed_forward_hidden, d_out=hidden_in,dropout=dropout)
#         self.input_sublayer_S = SublayerConnection(size=hidden_in, dropout=dropout)
#         self.output_sublayer_S = SublayerConnection(size=hidden_in, dropout=dropout)
#         # to do:  all frames share one attention map
#         # self.attention_S_context
#
#         # self-attetnion model on temporal dimension
#         self.attention_T = MultiHeadedAttention(h=attn_heads, d_model=hidden_in)
#         self.feed_forward_T = PositionwiseFeedForward(d_in=hidden_in, d_ff=feed_forward_hidden, d_out=hidden_out, dropout=dropout)
#         self.input_sublayer_T = SublayerConnection(size=hidden_in, dropout=dropout)
#         self.output_sublayer_T = SublayerConnection(size=hidden_out, dropout=dropout)
#         # to do: all joints share one attention map
#         # self.attention_T_context
#
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x, R_A, mask=None):
#         '''
#         R_A means the global regularized adjacency
#         mask is used for items with no token in the Bert model and is not necessary in our model
#         '''
#
#         # self-attetnion on spatial dimension
#
#         N, T, V, C = x.size()
#         x = x.view(N*T, V, C)
#         x = self.input_sublayer_S(x=x, sublayer=lambda _x, A_G: self.attention_S.forward(_x, _x, _x, mask=mask, nomal_A=A_G), A=R_A)
#         x = self.output_sublayer_S(x, self.feed_forward_S)
#
#         # self-attention on temporal dimension
#         C = x.size(-1)
#         x = x.view(N, T, V, C).permute(0,2,1,3).contiguous().view(N*V, T, C)
#         x = self.input_sublayer_T(x=x, sublayer=lambda _x: self.attention_T.forward(_x, _x, _x, mask=mask))
#         x = self.output_sublayer_T(x, self.feed_forward_T)
#         C = x.size(-1)
#
#         x = x.view(N, V, T, C).permute(0,2,1,3).contiguous().view(N, T, V, C)
#         return self.dropout(x)