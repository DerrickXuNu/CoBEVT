"""
CAV Fusion Transformer Baseline, partially migrated from vit-pytorch
"""
import os

import torch
import numpy as np
import math
from torch import nn

from einops import rearrange


class RelTemporalEncoding(nn.Module):
    """
    Implement the Temporal Encoding (Sinusoid) function.
    """

    def __init__(self, n_hid, RTE_ratio, max_len=100, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(
            n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(
            n_hid)
        emb.requires_grad = False
        self.RTE_ratio = RTE_ratio
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        # When t has unit of 50ms, rte_ratio=1. So we can train on 100ms but test on 50ms
        return x + self.lin(self.emb(t * self.RTE_ratio)).unsqueeze(
            0).unsqueeze(1)


class RTE(nn.Module):
    def __init__(self, dim, RTE_ratio=2):
        super(RTE, self).__init__()
        self.RTE_ratio = RTE_ratio

        self.emb = RelTemporalEncoding(dim, RTE_ratio=self.RTE_ratio)

    def forward(self, x, dts):
        # x: (B,L,H,W,C)
        # dts: (B,L)
        rte_batch = []
        for b in range(x.shape[0]):
            rte_list = []
            for i in range(x.shape[1]):

                rte_list.append(
                    self.emb(x[b, i, :, :, :], dts[b, i]).unsqueeze(0))
            rte_batch.append(torch.cat(rte_list, dim=0).unsqueeze(0))
        return torch.cat(rte_batch, dim=0)


class CavPositionalEncoding(nn.Module):

    def __init__(self, d_hid, cav_num=5):
        super(CavPositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table',
                             self._get_sinusoid_encoding_table(cav_num, d_hid))

    def _get_sinusoid_encoding_table(self, cav_num, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for
                    hid_j in range(d_hid)]

        # L, C
        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(cav_num)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)

    def forward(self, x):
        pos_table = self.pos_table[None, :, None, None, :].clone().detach()
        return x + pos_table


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CavAttention(nn.Module):
    def __init__(self, dim, heads, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask, prior_encoding=None):
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        # mask: (B, L)
        x = x.permute(0, 2, 3, 1, 4)
        # mask: (B, 1, H, W, 1, L)
        mask = mask.unsqueeze(1)

        # qkv: [(B, H, W, L, C_inner) *3]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c',
                                          m=self.heads), qkv)

        # attention, (B, M, H, W, L, L)
        att_map = torch.einsum('b m h w i c, b m h w j c -> b m h w i j',
                               q, k) * self.scale
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float('inf'))
        # softmax
        att_map = self.attend(att_map)

        # out:(B, M, H, W, L, C_head)
        out = torch.einsum('b m h w i j, b m h w j c -> b m h w i c', att_map,
                           v)
        out = rearrange(out, 'b m h w l c -> b h w l (m c)',
                        m=self.heads)
        out = self.to_out(out)
        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)
        return out


class HGTCavAttention(nn.Module):
    def __init__(self, dim, heads, num_types=2,
                 num_relations=4, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_types = num_types

        self.attend = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(dropout)
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for t in range(num_types):
            self.k_linears.append(nn.Linear(dim, inner_dim))
            self.q_linears.append(nn.Linear(dim, inner_dim))
            self.v_linears.append(nn.Linear(dim, inner_dim))
            self.a_linears.append(nn.Linear(inner_dim, dim))

        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations, heads, dim_head, dim_head))
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations, heads, dim_head, dim_head))

        torch.nn.init.xavier_uniform(self.relation_att)
        torch.nn.init.xavier_uniform(self.relation_msg)

    def to_qkv(self, x, types):
        # x: (B,H,W,L,C)
        # types: (B,L)
        q_batch = []
        k_batch = []
        v_batch = []

        for b in range(x.shape[0]):
            q_list = []
            k_list = []
            v_list = []

            for i in range(x.shape[-2]):
                # (H,W,1,C)
                q_list.append(
                    self.q_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
                k_list.append(
                    self.k_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
                v_list.append(
                    self.v_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
            # (1,H,W,L,C)
            q_batch.append(torch.cat(q_list, dim=2).unsqueeze(0))
            k_batch.append(torch.cat(k_list, dim=2).unsqueeze(0))
            v_batch.append(torch.cat(v_list, dim=2).unsqueeze(0))
        # (B,H,W,L,C)
        q = torch.cat(q_batch, dim=0)
        k = torch.cat(k_batch, dim=0)
        v = torch.cat(v_batch, dim=0)
        return q, k, v

    def get_relation_type_index(self, type1, type2):
        return type1 * self.num_types + type2

    def get_hetero_edge_weights(self, x, types):
        w_att_batch = []
        w_msg_batch = []

        for b in range(x.shape[0]):
            w_att_list = []
            w_msg_list = []

            for i in range(x.shape[-2]):
                w_att_i_list = []
                w_msg_i_list = []

                for j in range(x.shape[-2]):
                    e_type = self.get_relation_type_index(types[b, i],
                                                          types[b, j])
                    w_att_i_list.append(self.relation_att[e_type].unsqueeze(0))
                    w_msg_i_list.append(self.relation_msg[e_type].unsqueeze(0))
                w_att_list.append(torch.cat(w_att_i_list, dim=0).unsqueeze(0))
                w_msg_list.append(torch.cat(w_msg_i_list, dim=0).unsqueeze(0))

            w_att_batch.append(torch.cat(w_att_list, dim=0).unsqueeze(0))
            w_msg_batch.append(torch.cat(w_msg_list, dim=0).unsqueeze(0))

        # (B,M,L,L,C_head,C_head)
        w_att = torch.cat(w_att_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        w_msg = torch.cat(w_msg_batch, dim=0).permute(0, 3, 1, 2, 4, 5)
        return w_att, w_msg

    def to_out(self, x, types):
        out_batch = []
        for b in range(x.shape[0]):
            out_list = []
            for i in range(x.shape[-2]):
                out_list.append(
                    self.a_linears[types[b, i]](x[b, :, :, i, :].unsqueeze(2)))
            out_batch.append(torch.cat(out_list, dim=2).unsqueeze(0))
        out = torch.cat(out_batch, dim=0)
        return out

    def forward(self, x, mask, prior_encoding):
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        # mask: (B, H, W, L, 1)
        # prior_encoding: (B,L,H,W,3)
        x = x.permute(0, 2, 3, 1, 4)
        # mask: (B, 1, H, W, L, 1)
        mask = mask.unsqueeze(1)
        # (B,L)
        velocities, dts, types = [itm.squeeze(-1) for itm in
                                  prior_encoding[:, :, 0, 0, :].split(
                                      [1, 1, 1], dim=-1)]
        types = types.to(torch.int)
        dts = dts.to(torch.int)
        qkv = self.to_qkv(x, types)
        # (B,M,L,L,C_head,C_head)
        w_att, w_msg = self.get_hetero_edge_weights(x, types)

        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c',
                                          m=self.heads), (qkv))
        # attention, (B, M, H, W, L, L)
        att_map = torch.einsum(
            'b m h w i p, b m i j p q, bm h w j q -> b m h w i j',
            [q, w_att, k]) * self.scale
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float('inf'))
        # softmax
        att_map = self.attend(att_map)

        # out:(B, M, H, W, L, C_head)
        v_msg = torch.einsum('b m i j p c, b m h w j p -> b m h w i j c',
                             w_msg, v)
        out = torch.einsum('b m h w i j, b m h w i j c -> b m h w i c',
                           att_map, v_msg)

        out = rearrange(out, 'b m h w l c -> b h w l (m c)',
                        m=self.heads)
        out = self.to_out(out, types)
        out = self.drop_out(out)
        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)
        return out


class BaseEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CavAttention(dim,
                                          heads=heads,
                                          dim_head=dim_head,
                                          dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class BaseTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        dim = args['dim']
        depth = args['depth']
        heads = args['heads']
        dim_head = args['dim_head']
        mlp_dim = args['mlp_dim']
        dropout = args['dropout']

        self.encoder = BaseEncoder(dim, depth, heads, dim_head, mlp_dim,
                                   dropout)

    def forward(self, x, mask):
        # B, L, H, W, C
        output = self.encoder(x, mask)
        # B, H, W, C
        output = output[:, 0]

        return output


if __name__ == "__main__":
    # test transformer
    args = {'dim': 256,
            'depth': 3,
            'heads': 8,
            'dim_head': 32,
            'mlp_dim': 256,
            'dropout': 0.2,
            'max_cav': 5}

    transformer = BaseTransformer(args)
    transformer.cuda()

    x = torch.randn(1, 5, 176, 96, 256)
    x = x.cuda()

    mask_ = torch.from_numpy(np.array([[1, 1, 1, 0, 0]], dtype=int))
    mask_ = mask_.cuda()
    output = transformer(x, mask_)

    loss_func = nn.MSELoss()
    loss = loss_func(x[:, 0], output)
    loss.backward()

    print(loss)
