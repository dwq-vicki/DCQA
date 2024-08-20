# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless requreader2d by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

#ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class DPDALayear(nn.Module):
    def __init__(self, dim):
        super(DPDALayear, self).__init__()
        self.W_p = nn.Linear(2 * dim, dim)
        self.W_q = nn.Linear(2 * dim, dim)
        self.W_map = nn.Linear(dim, dim)


    def forward(self, P, Q, p_mask=None, q_mask=None):
        # P = self.W_map(P)
        P_ori = P
        Q_ori = Q
        A = torch.matmul(P, Q.transpose(dim0=1, dim1=2))  # batch, l_p, l_q

        if p_mask is not None:
            p_mask = p_mask.float()
            p_mask = 1 - p_mask
            p_mask = p_mask * -10000.0
            p_mask = p_mask.unsqueeze(dim=2)
            p_mask = p_mask.expand_as(A)
            A = A + p_mask
            A = A.to(P.dtype)

        if q_mask is not None:
            q_mask = q_mask.float()
            q_mask = 1 - q_mask
            q_mask = q_mask * -10000.0
            q_mask = q_mask.unsqueeze(dim=1)
            q_mask = q_mask.expand_as(A)
            A = A + q_mask
            A = A.to(Q.dtype)

        # weight matrix
        p_weight, _ = torch.max(A, dim=2)
        q_weight, _ = torch.max(A, dim=1)
        # if p_mask is not None:
        #     p_weight *= p_mask
        # if q_mask is not None:
        #     q_weight *= q_mask

        A_q = torch.softmax(A, dim=2)  # l_p, l_q
        A_p = torch.softmax(A.transpose(dim0=1, dim1=2), dim=2)  # l_q, l_p

        P_q = torch.matmul(A_q, Q)  # l_p, dim
        Q_p = torch.matmul(A_p, P)  # l_q, dim

        P_t = torch.cat([P_q, P], dim=2)  # l_p, 2*dim
        Q_t = torch.cat([Q_p, Q], dim=2)  # l_q, 2*dim

        Q = torch.matmul(A_p, P_t)  # l_q, 2*dim
        P = torch.matmul(A_q, Q_t)  # l_p, 2*dim
        P = P_ori + self.W_p(P)  # l_p, dim
        layer_norm = nn.LayerNorm(normalized_shape=[P.size(-2), P.size(-1)],elementwise_affine=False)
        P = layer_norm(P)
        Q = Q_ori + self.W_q(Q)  # l_q, dim
        layer_norm = nn.LayerNorm(normalized_shape=[Q.size(-2), Q.size(-1)],elementwise_affine=False)
        Q = layer_norm(Q)
        return P, Q, p_weight, q_weight

# cross_attention
class Crossattention(nn.Module):
    def __init__(self, dim, layer_num):
        super(Crossattention, self).__init__()
        dpda_layer = DPDALayear(dim)
        self.q_o = nn.ModuleList([copy.deepcopy(dpda_layer) for _ in range(layer_num)])
        '''
        self.linear = nn.Linear(2 * dim, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.linear = nn.Linear(2 * dim, dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        '''

    def forward(self, q, o, mask_q, mask_o):
        for layer_module in self.q_o:
            Q, O, q_weight, o_weight = layer_module(q, o, mask_q, mask_o)
        q, _ = torch.max(Q, dim=1)
        o, _ = torch.max(O, dim=1)
        # q_o = self.get_vector(q, o)
        return q, o, Q, O, q_weight, o_weight
        # return q_o, q_weight, o_weight


def get_weight(v1, v2, W, b):
    """
    v1: (b, l1, h)
    v2: (b, l2, h)
    W: (h, h)
    b: (1, )
    return (b, l1, l2)
    """

    # print("hidden is {}".format(v1.size()))
    # print("result1 is {}".format((v1 @ W).tolist()[0][0][:5]))
    # print("result2 is {}".format(v2.permute(0, 2, 1).tolist()[0][0][:5]))
    # print("result3 is {}".format((v1 @ W).matmul(v2.permute(0, 2, 1)).tolist()[0][0][:5]))
    # print("size is {}".format(torch.matmul((v1 @ W),v2.permute(0, 2, 1)).size()))
    # return torch.matmul((v1 @ W),v2.permute(0, 2, 1)) + b
    return (v1 @ W).bmm(v2.permute(0, 2, 1)) + b
    # return (v1 @ W + b).bmm(v2.permute(0, 2, 1))
# csqa
class ChoiceAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ChoiceAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W.data.normal_(mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.normal_(mean=0.0, std=0.02)

    def retain(self, option1, option2, option3, option4, option5):
        """
        option1, (b, s1, h)
        option2 option3, (b, s2, h)
        """
        weight2 = get_weight(option1, option2, self.W, self.bias).unsqueeze(-1)
        weight3 = get_weight(option1, option3, self.W, self.bias).unsqueeze(-1)
        weight4 = get_weight(option1, option4, self.W, self.bias).unsqueeze(-1)
        weight5 = get_weight(option1, option5, self.W, self.bias).unsqueeze(-1)
        weight = torch.cat((weight2, weight3, weight4, weight5), dim=-1)  # (b, s1, s2, 2)
        weight = torch.softmax(weight, dim=-1)
        option1_ = weight[:, :, :, 0] @ option2 + weight[:, :, :, 1] @ option3 + weight[:, :, :, 2] @ option4 +weight[:, :, :, 3] @ option5
        return option1_

    def forward(self, option1, option2, option3, option4, option5):
        option1_ = self.retain(option1, option2, option3, option4, option5)
        option2_ = self.retain(option2, option1, option3, option4, option5)
        option3_ = self.retain(option3, option1, option2, option4, option5)
        option4_ = self.retain(option4, option1, option2, option3, option5)
        option5_ = self.retain(option5, option1, option2, option3, option4)

        #option1_ = option1 - option1_
        #option2_ = option2 - option2_
        #option3_ = option3 - option3_
        similar = (option1_ + option2_ + option3_ + option4_ + option5_)/2

        #return option1_, option2_, option3_, option4_, option5_
        return similar
'''

# qasc
class ChoiceAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ChoiceAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W.data.normal_(mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.normal_(mean=0.0, std=0.02)

    def retain(self, option1, option2, option3, option4, option5, option6, option7, option8):
        """
        option1, (b, s1, h)
        option2 option3, (b, s2, h)
        """
        weight2 = get_weight(option1, option2, self.W, self.bias).unsqueeze(-1)
        weight3 = get_weight(option1, option3, self.W, self.bias).unsqueeze(-1)
        weight4 = get_weight(option1, option4, self.W, self.bias).unsqueeze(-1)
        weight5 = get_weight(option1, option5, self.W, self.bias).unsqueeze(-1)
        weight6 = get_weight(option1, option6, self.W, self.bias).unsqueeze(-1)
        weight7 = get_weight(option1, option7, self.W, self.bias).unsqueeze(-1)
        weight8 = get_weight(option1, option8, self.W, self.bias).unsqueeze(-1)
        weight = torch.cat((weight2, weight3, weight4, weight5, weight6, weight7, weight8), dim=-1)  # (b, s1, s2, 2)
        weight = torch.softmax(weight, dim=-1)
        option1_ = weight[:, :, :, 0] @ option2 + weight[:, :, :, 1] @ option3 + weight[:, :, :, 2] @ option4 +weight[:, :, :, 3] @ option5 + weight[:, :, :, 4] @ option6 + weight[:, :, :, 5] @ option7 + weight[:, :, :, 6] @ option8
        return option1_

    def forward(self, option1, option2, option3, option4, option5, option6, option7, option8):
        option1_ = self.retain(option1, option2, option3, option4, option5, option6, option7, option8)
        option2_ = self.retain(option2, option1, option3, option4, option5, option6, option7, option8)
        option3_ = self.retain(option3, option1, option2, option4, option5, option6, option7, option8)
        option4_ = self.retain(option4, option1, option2, option3, option5, option6, option7, option8)
        option5_ = self.retain(option5, option1, option2, option3, option4, option6, option7, option8)
        option6_ = self.retain(option6, option1, option2, option3, option4, option5, option7, option8)
        option7_ = self.retain(option7, option1, option2, option3, option4, option5, option6, option8)
        option8_ = self.retain(option8, option1, option2, option3, option4, option5, option6, option7)

        # option1_ = option1 - option1_
        # option2_ = option2 - option2_
        # option3_ = option3 - option3_

        similar = (option1_ + option2_ + option3_ + option4_ + option5_ + option6_ + option7_ + option8_) / 2

        # return option1_, option2_, option3_, option4_, option5_
        return similar
'''
'''
# socialiqa
class ChoiceAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ChoiceAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W.data.normal_(mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.normal_(mean=0.0, std=0.02)

    def retain(self, option1, option2, option3):
        """
        option1, (b, s1, h)
        option2 option3, (b, s2, h)
        """
        weight2 = get_weight(option1, option2, self.W, self.bias).unsqueeze(-1)
        weight3 = get_weight(option1, option3, self.W, self.bias).unsqueeze(-1)
        weight = torch.cat((weight2, weight3), dim=-1)  # (b, s1, s2, 2)
        weight = torch.softmax(weight, dim=-1)
        option1_ = weight[:, :, :, 0] @ option2 + weight[:, :, :, 1] @ option3
        return option1_

    def forward(self, option1, option2, option3):
        option1_ = self.retain(option1, option2, option3)
        option2_ = self.retain(option2, option1, option3)
        option3_ = self.retain(option3, option1, option2)

        #option1_ = option1 - option1_
        #option2_ = option2 - option2_
        #option3_ = option3 - option3_

        similar = (option1_ + option2_ + option3_)/2

        #return option1_, option2_, option3_, option4_, option5_
        return similar
'''
'''
# obqa, arc-easy, arc-challenge
class ChoiceAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ChoiceAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W.data.normal_(mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.normal_(mean=0.0, std=0.02)

    def retain(self, option1, option2, option3, option4):
        """
        option1, (b, s1, h)
        option2 option3, (b, s2, h)
        """
        weight2 = get_weight(option1, option2, self.W, self.bias).unsqueeze(-1)
        weight3 = get_weight(option1, option3, self.W, self.bias).unsqueeze(-1)
        weight4 = get_weight(option1, option4, self.W, self.bias).unsqueeze(-1)
        weight = torch.cat((weight2, weight3, weight4), dim=-1)  # (b, s1, s2, 2)
        weight = torch.softmax(weight, dim=-1)
        option1_ = weight[:, :, :, 0] @ option2 + weight[:, :, :, 1] @ option3 + weight[:, :, :, 2] @ option4
        return option1_

    def forward(self, option1, option2, option3, option4):
        option1_ = self.retain(option1, option2, option3, option4)
        option2_ = self.retain(option2, option1, option3, option4)
        option3_ = self.retain(option3, option1, option2, option4)
        option4_ = self.retain(option4, option1, option2, option3)

        #option1_ = option1 - option1_
        #option2_ = option2 - option2_
        #option3_ = option3 - option3_

        similar = (option1_ + option2_ + option3_ + option4_)/2

        #return option1_, option2_, option3_, option4_, option5_
        return similar
'''
'''
# piqa
class ChoiceAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ChoiceAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W.data.normal_(mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.bias.data.normal_(mean=0.0, std=0.02)

    def retain(self, option1, option2):
        """
        option1, (b, s1, h)
        option2 option3, (b, s2, h)
        """
        weight2 = get_weight(option1, option2, self.W, self.bias).unsqueeze(-1)
        weight = weight2  # (b, s1, s2, 2)
        weight = torch.softmax(weight, dim=-1)
        option1_ = weight[:, :, :, 0] @ option2
        return option1_

    def forward(self, option1, option2):
        option1_ = self.retain(option1, option2)
        option2_ = self.retain(option2, option1)

        #option1_ = option1 - option1_
        #option2_ = option2 - option2_
        #option3_ = option3 - option3_

        similar = (option1_ + option2_)/2

        #return option1_, option2_, option3_, option4_, option5_
        return similar

'''

class Self_attention(nn.Module):
    """
    H (B, L, hidden_size) => h (B, hidden_size)
    """
    def __init__(self, input_size, attention_size, dropout_prob):
        super(Self_attention, self).__init__()
        self.attention_size = attention_size
        self.hidden_layer = nn.Linear(input_size, self.attention_size)
        self.query_ = nn.Parameter(torch.Tensor(self.attention_size, 1))
        self.dropout = nn.Dropout(dropout_prob)

        self.query_.data.normal_(mean=0.0, std=0.02)

    def forward(self, values, mask=None):
        """
        (b, l, h) -> (b, h)
        """
        if mask is None:
            mask = torch.zeros_like(values)
            # mask = mask.data.normal_(mean=0.0, std=0.02)
        else:
            mask = (1 - mask.unsqueeze(-1).type(torch.float)) * -1000.

        keys = self.hidden_layer(values)
        keys = torch.tanh(keys)
        query_var = torch.var(self.query_)
        # (b, l, h) + (h, 1) -> (b, l, 1)
        attention_probs = keys @ self.query_ / math.sqrt(self.attention_size * query_var)
        # attention_probs = keys @ self.query_ / math.sqrt(self.attention_size)

        attention_probs = torch.softmax(attention_probs * mask, dim=1)
        attention_probs = self.dropout(attention_probs)

        context = torch.sum(attention_probs + values, dim=1)
        return context