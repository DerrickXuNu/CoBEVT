import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from .encoder import *


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


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


# swap attention -> max_vit
class Attention(nn.Module):
    """
    Unit Attention class. Todo: mask is not added yet.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        The agent can be different views, timestamps or vehicles.
    """

    def __init__(
            self,
            dim,
            head=8,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7
    ):
        super().__init__()

        self.heads = head
        inner_dim = head * dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x):
        # x shape: b, l, h, w, w_h, w_w, c
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)


class SwapFusionBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention.
    """

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 head,
                 dim_head,
                 window_size,
                 agent_size,
                 depth,
                 drop_out):
        super(SwapFusionBlock, self).__init__()
        # b = batch * max_cav
        self.depth = depth
        self.layers = nn.ModuleList([])

        for i in range(self.depth):
            block = nn.Sequential(
                Rearrange('b m d (x w1) (y w2) -> b m x y w1 w2 d',
                          w1=window_size, w2=window_size),
                PreNormResidual(input_dim,
                                Attention(input_dim, head, dim_head, drop_out,
                                          agent_size, window_size)),
                PreNormResidual(input_dim,
                                FeedForward(input_dim, mlp_dim, drop_out)),
                Rearrange('b m x y w1 w2 d -> b m d (x w1) (y w2)'),

                Rearrange('b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                          w1=window_size, w2=window_size),
                PreNormResidual(input_dim,
                                Attention(input_dim, head, dim_head, drop_out,
                                          agent_size, window_size)),
                PreNormResidual(input_dim,
                                FeedForward(input_dim, mlp_dim, drop_out)),
                Rearrange('b m x y w1 w2 d -> b m d (w1 x) (w2 y)'),
            )
            self.layers.append(block)

    def forward(self, x):
        # x shape: b, n, d, h, w
        for stage in self.layers:
            x = stage(x)
        return x


class SwapFusionModule(nn.Module):
    """
    Swap Fusion Module contains series of swap fusion blocks.
    """

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 head,
                 dim_head,
                 window_size,
                 agent_size,
                 depth,
                 drop_out):
        super(SwapFusionModule, self).__init__()
        assert len(input_dim) == len(mlp_dim) == len(head) == \
               len(window_size) == len(agent_size) == len(depth) == \
               len(drop_out)

        self.iterations = len(input_dim)
        self.layers = nn.ModuleList([])

        for i in range(self.iterations):
            layer = SwapFusionBlock(input_dim[i],
                                    mlp_dim[i],
                                    head[i],
                                    dim_head[i],
                                    window_size[i],
                                    agent_size[i],
                                    depth[i],
                                    drop_out[i])
            self.layers.append(layer)

    def forward(self, x, i):
        output = self.layers[i](x)
        return output


class SwapEncoder(nn.Module):
    def __init__(
            self,
            backbone,
            swap_module,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone
        self.swap_module = swap_module

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale,
                                                recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(
                torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim,
                                     **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(
                *[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)  # b n c h w
        I_inv = batch['intrinsics'].inverse()  # b n 3 3
        E_inv = batch['extrinsics'].inverse()  # b n 4 4

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()  # d H W
        x = repeat(x, '... -> b ...', b=b)  # b d H W

        for i, (cross_view, feature, layer) in enumerate(zip(self.cross_views,
                                                             features,
                                                             self.layers)):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            feature = self.swap_module(feature, i)

            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)

        return x


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    block = SwapFusionModule([32, 512], [128, 512], [8, 32], [4, 8], [6, 6],
                             [2, 2], [0, 0])
    block.cuda()
    test_data = torch.rand(1, 6, 32, 56, 120)
    test_data = test_data.cuda()

    output = block(test_data, 0)
    print(output.shape)
