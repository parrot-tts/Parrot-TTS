# feed forward transformer

import torch
from torch import nn
import torch.nn.functional as F

import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        pe_table = self.positionalencoding1d(d_model, max_len)
        self.register_buffer('pe', pe_table)

    def forward(self, x):
        x = self.pe[x.size(1)] + x 
        return x

    @staticmethod
    def positionalencoding1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
    
        return pe 

        
class Attention(nn.Module):
    def __init__(self, d_model, n_head, dropout_p=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model 
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout_p, bias=False, batch_first=True)
        self.wo  = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout_p

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        bsz, seqlen, _ = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=2)
        y, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        y = self.wo(y)

        return y
        

class ConvLayer(nn.Module):
    def __init__(self, d_model, n_filter, kernel_sizes):
        super().__init__()
        self.conv1 = nn.Conv1d(
            d_model, 
            n_filter,
            kernel_size=kernel_sizes[0],
            padding=(kernel_sizes[0] - 1 ) // 2
        )
        self.conv2 = nn.Conv1d(
            n_filter, 
            d_model,
            kernel_size=kernel_sizes[1],
            padding=(kernel_sizes[1] - 1 ) // 2
        )

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.conv2(F.relu(self.conv1(output)))
        output = output.transpose(1,2)
        return output
        
        
class FFTBlock(nn.Module):
    def __init__(self, d_model, n_head, n_filter, kernel_sizes, dropout_p):
        super().__init__()
        self.d_model = d_model
        self.attention = Attention(d_model, n_head, dropout_p)
        self.convlayer = ConvLayer(d_model, n_filter, kernel_sizes)
        self.attn_norm = nn.LayerNorm(d_model)
        self.conv_norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # key_padding_mask: For a binary mask, a True value indicates that the 
        # corresponding key value will be ignored for the purpose of attention.
        # attn_mask: works similar to key_padding_mask
        h = x + self.attention(self.attn_norm(x), key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        out = h + self.convlayer(self.conv_norm(h))
        return out