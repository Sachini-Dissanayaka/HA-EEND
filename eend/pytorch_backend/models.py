# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import ModuleList
from torch.nn import MultiheadAttention
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import LayerNorm

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


class NoamScheduler(_LRScheduler):
    """
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Args:
        d_model: int
            The number of expected features in the encoder inputs.
        warmup_steps: int
            The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

        # the initial learning rate is set as step = 1
        if self.last_epoch == -1:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
            self.last_epoch = 0
        print(self.d_model)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

class MHLocalDenseSynthesizerAttention(nn.Module):
    """Multi-Head Local Dense Synthesizer attention layer
    In this implementation, the calculation of multi-head mechanism is similar to that of self-attention,
    but it takes more time for training. We provide an alternative multi-head mechanism implementation
    that can achieve competitive results with less time.
    :param int n_head: the number of heads
    :param int n_feat: the dimension of features
    :param float dropout_rate: dropout rate
    :param int context_size: context size
    :param bool use_bias: use bias term in linear layers
    """

    def __init__(self, n_head, n_feat, dropout_rate, context_size=15, use_bias=False):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.c = context_size
        self.w1 = nn.Linear(n_feat, n_feat, bias=use_bias)
        # self.w2 = nn.Linear(n_feat, n_head * self.c, bias=use_bias)
        self.w2 = nn.Conv1d(in_channels=n_feat, out_channels=n_head * self.c, kernel_size=1,
                            groups=n_head)
        self.w3 = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.w_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Forward pass.
                :param torch.Tensor query: (batch, time, size)
                :param torch.Tensor key: (batch, time, size) dummy
                :param torch.Tensor value: (batch, time, size)
                :param torch.Tensor mask: (batch, time, time) dummy
                :return torch.Tensor: attentioned and transformed `value` (batch, time, d_model)
                """
        bs, time = query.size()[: 2]
        query = self.w1(query)  # [B, T, d]
        # [B, T, d] --> [B, d, T] --> [B, H*c, T]
        weight = self.w2(torch.relu(query).transpose(1, 2))
        # [B, H, c, T] --> [B, T, H, c] --> [B*T, H, 1, c]
        weight = weight.view(bs, self.h, self.c, time).permute(0, 3, 1, 2) \
            .contiguous().view(bs * time, self.h, 1, self.c)
        value = self.w3(value)  # [B, T, d]
        # [B*T, c, d] --> [B*T, c, H, d_k] --> [B*T, H, c, d_k]
        value_cw = chunkwise(value, (self.c - 1) // 2, 1, (self.c - 1) // 2) \
            .view(bs * time, self.c, self.h, self.d_k).transpose(1, 2)
        self.attn = torch.softmax(weight, dim=-1)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value_cw)
        x = x.contiguous().view(bs, -1, self.h * self.d_k)  # [B, T, d]
        x = self.w_out(x)  # [B, T, d]
        return x

def chunkwise(xs, N_l, N_c, N_r):
    """Slice input frames chunk by chunk.
    Args:
        xs (FloatTensor): `[B, T, input_dim]`
        N_l (int): number of frames for left context
        N_c (int): number of frames for current context
        N_r (int): number of frames for right context
    Returns:
        xs (FloatTensor): `[B * n_chunks, N_l + N_c + N_r, input_dim]`
            where n_chunks = ceil(T / N_c)
    """
    bs, xmax, idim = xs.size()
    n_chunks = math.ceil(xmax / N_c)
    c = N_l + N_c + N_r
    s_index = torch.arange(0, xmax, N_c).unsqueeze(-1)
    c_index = torch.arange(0, c)
    index = s_index + c_index
    xs_pad = torch.cat([xs.new_zeros(bs, N_l, idim),
                        xs,
                        xs.new_zeros(bs, N_c*n_chunks-xmax+N_r, idim)], dim=1)
    xs_chunk = xs_pad[:, index].contiguous().view(bs * n_chunks, N_l + N_c + N_r, idim)
    return xs_chunk


class MultiHeadExternalAttention(nn.Module):

    def __init__(self, n_head, n_feat, S=64):
        super().__init__()
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.mk = nn.Linear(n_feat, S, bias=False)
        self.mv = nn.Linear(S, n_feat, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.d_k = n_feat // n_head
        self.h = n_head

    def forward(self, query):

        n_batch = query.size(0)
        
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k) # B, T, D --> B, T, H, d_k
        q = q.transpose(1, 2)  # (B, H, T, d_k)

        attn = self.mk(q) # B, H, T, S
        attn = self.softmax(attn) # B, H, T, S
        attn = attn/torch.sum(attn, dim=3, keepdim=True) # B, H, T, S
        out = self.mv(attn) # B, H, T, d_k

        out = out.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # B, T, D
        return self.linear_out(x)  # B, T, D

    
class LocalDenseSynthesizerAttention(nn.Module):
    """Multi-Head Local Dense Synthesizer attention layer
    
    :param int n_head: the number of heads
    :param int n_feat: the dimension of features
    :param float dropout_rate: dropout rate
    :param int context_size: context size
    :param bool use_bias: use bias term in linear layers

    """
    def __init__(self, n_head, n_feat, dropout_rate, context_size=30, use_bias=False):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.c = context_size
        self.w1 = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.w2 = nn.Linear(n_feat, n_head*self.c, bias=use_bias)
        self.w3 = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.w_out = nn.Linear(n_feat, n_feat, bias=use_bias)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Forward pass.
        :param torch.Tensor query: (batch, time, size)
        :param torch.Tensor key: (batch, time, size) dummy
        :param torch.Tensor value: (batch, time, size)
        :param torch.Tensor mask: (batch, time, time) dummy
        :return torch.Tensor: attentioned and transformed `value` (batch, time, d_model)
        """
        bs, time = query.size()[: 2]
        query = self.w1(query)  # [B, T, d]
        # [B, T, H*c] --> [B, T, H, c] --> [B, H, T, c]
        weight = self.w2(torch.relu(query)).view(bs, time, self.h, self.c).transpose(1, 2).contiguous()

        scores = torch.zeros(bs * self.h * time * (time + self.c - 1), dtype=weight.dtype)
        scores = scores.view(bs, self.h, time, time + self.c - 1).fill_(float("-inf"))
        scores = scores.to(query.device)  # [B, H, T, T+c-1]
        scores.as_strided(
            (bs, self.h, time, self.c),
            ((time + self.c - 1) * time * self.h, (time + self.c - 1) * time, time + self.c, 1)
        ).copy_(weight)
        scores = scores.narrow(-1, int((self.c - 1) / 2), time)  # [B, H, T, T]
        self.attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(self.attn)

        value = self.w3(value).view(bs, time, self.h, self.d_k)  # [B, T, H, d_k]
        value = value.transpose(1, 2).contiguous()  # [B, H, T, d_k]
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(bs, time, self.h*self.d_k)
        x = self.w_out(x)  # [B, T, d]
        return x

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class HybridAttention(nn.Module):
    """Combination of MHSA and LDSA
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    :param int context_size: context size
    """

    def __init__(self, n_head, n_feat, dropout_rate, dim_feedforward, context_size=30):
        super(HybridAttention, self).__init__()
        
        # Attention modules
        self.self_att = MultiHeadedAttention(n_head, n_feat, dropout_rate)
        # self.ldsa_att = LocalDenseSynthesizerAttention(n_head, n_feat, dropout_rate, context_size)
        
        # Implementation of Position-wise Feed-Forward model
        self.linear1 = Linear(n_feat, dim_feedforward) 
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.linear2 = Linear(dim_feedforward, n_feat)

        # Normalization layers
        self.norm1 = LayerNorm(n_feat)
        self.norm2 = LayerNorm(n_feat)
        self.norm3 = LayerNorm(n_feat)

        # Dropout layers
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)


    def forward(self, q, k, v, mask):
        """
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor: attentioned and transformed `value`
        """

        e = q

        # layer normalization
        e = self.norm1(e)
        # self-attention
        s = self.self_att(e, e, e, mask)
        # residual
        e = e + self.dropout1(s)

        # # layer normalization
        # e = self.norm2(e)
        # # local dense synthesizer attention
        # s = self.ldsa_att(e, e, e, mask)
        # # residual
        # e = e + self.dropout2(s)

        # layer normalization
        e = self.norm3(e)
        # positionwise feed-forward
        s = self.linear2(self.dropout(self.relu(self.linear1(e))))
        # residual
        e = e + self.dropout3(s)

        return e

class TransformerModel(nn.Module):
    def __init__(self, n_speakers, in_size, n_heads, n_units, n_layers, dim_feedforward=2048, dropout=0.5, has_pos=False):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(TransformerModel, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads #num of parallel layers
        self.n_units = n_units #num of nodes
        self.n_layers = n_layers 
        self.has_pos = has_pos

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        # self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        # encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        # ldsa_layer=LocalDenseSynthesizerAttention(n_heads, n_units, dropout)
        hybrid_layer = HybridAttention(n_heads, n_units, dropout, dim_feedforward)
        self.transformer_encoder = ModuleList([copy.deepcopy(hybrid_layer) for i in range(n_layers)])
        # self.transformer_encoder = HybridAttention(n_heads, n_units, dropout, dim_feedforward)
        # self.transformer_encoder = LocalDenseSynthesizerAttention(n_heads, n_units, dropout)
        self.norm_out = nn.LayerNorm(n_units)
        self.decoder = nn.Linear(n_units, n_speakers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=False, activation=None):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != srz.size(1):
                mask = self._generate_square_subsequent_mask(srz.size(1)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)

        # src: (B, T, E)
        src = self.encoder(src)

        # src = self.encoder_norm(src)

        # src: (T, B, E)
        # src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = src

        for mod in self.transformer_encoder:
            output = mod(output, output, output, self.src_mask)
        # output = self.transformer_encoder(src, src, src, self.src_mask)
        # output: (B, T, E)
        # output = output.transpose(0, 1)
        # output: (B, T, C)
        output = self.norm_out(output)
        output = self.decoder(output)

        if activation:
            output = activation(output)

        output = [out[:ilen] for out, ilen in zip(output, ilens)]

        return output

    def get_attention_weight(self, src):
        # NOTE: NOT IMPLEMENTED CORRECTLY!!!
        attn_weight = []
        def hook(module, input, output):
            # attn_output, attn_output_weights = multihead_attn(query, key, value)
            # output[1] are the attention weights
            attn_weight.append(output[1])
            
        handles = []
        for l in range(self.n_layers):
            handles.append(self.transformer_encoder.layers[l].self_attn.register_forward_hook(hook))

        self.eval()
        with torch.no_grad():
            self.forward(src)

        for handle in handles:
            handle.remove()
        self.train()

        return torch.stack(attn_weight)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


if __name__ == "__main__":
    import torch
    model = TransformerModel(5, 40, 4, 512, 2, 0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input).size())
    print("Model attention:", model.get_attention_weight(input).size())
    print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())