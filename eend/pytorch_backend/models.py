# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

from eend.pytorch_backend.convolution import Conv2dSubampling


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

class LocalDenseSynthesizerAttention(nn.Module):
    """Multi-Head Local Dense Synthesizer attention layer
    
    :param int n_head: the number of heads
    :param int n_feat: the dimension of features
    :param float dropout_rate: dropout rate
    :param int context_size: context size
    :param bool use_bias: use bias term in linear layers

    """
    def __init__(self, n_head, n_feat, dropout_rate, context_size=63, use_bias=False):
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
        return self.linear_out(x), self.attn  # (batch, time1, d_model)

class HybridAttention(nn.Module):
    """Combination of MHSA and LDSA
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    :param int context_size: context size
    """

    def __init__(self, n_head, n_feat, dropout_rate, context_size=78):
        super(HybridAttention, self).__init__()
        self.dot_att = MultiHeadedAttention(n_head, n_feat, dropout_rate)
        self.ldsa_att = LocalDenseSynthesizerAttention(n_head, n_feat, dropout_rate, context_size)

    def forward(self, query, key, value, mask):
        """
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :return torch.Tensor: attentioned and transformed `value`
        """
        x = self.ldsa_att(query, key, value, mask)
        x = self.dot_att(x, x, x, mask)
        return x

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
        # self.conv_subsampling = Conv2dSubampling(in_channels = 1, out_channels = 1)
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        # encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.transformer_encoder = HybridAttention(n_heads, n_units, dropout)
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
        # print("before",src.size())
        # src = self.conv_subsampling(src,src.size())
        # print("after",src.size())
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        # src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # output: (T, B, E)
        output = self.transformer_encoder(src, src, src, self.src_mask)[0]
        # output: (B, T, E)
        # output = output.transpose(0, 1)
        # output: (B, T, C)
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
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    import torch
    model = TransformerModel(5, 40, 4, 512, 2, 0.1)
    input = torch.randn(8, 500, 40)
    print("Model output:", model(input).size())
    print("Model attention:", model.get_attention_weight(input).size())
    print("Model attention sum:", model.get_attention_weight(input)[0][0][0].sum())
