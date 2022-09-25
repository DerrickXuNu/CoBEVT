"""
Implementation of V2VNet Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine, get_rotated_roi
from opencood.models.sub_modules.convgru import ConvGRU


class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1,
                                 padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x, mask=None):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))
        if mask is not None:
            x_1 = x_1.masked_fill(mask == 0, -float('inf'))
        return self.softmax(x_1)


class DiscoNetFusion(nn.Module):
    def __init__(self, args):
        super(DiscoNetFusion, self).__init__()
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W']
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']

        self.use_temporal_encoding = args['use_temporal_encoding']
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']
        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']
        self.agg_operator = args['agg_operator']
        self.use_mask = args['use_mask']

        self.cnn = nn.Conv2d(in_channels + 1, in_channels, kernel_size=3,
                             stride=1, padding=1)
        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels],
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)
        self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(in_channels)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, pairwise_t_matrix, prior_encoding=None):
        # x: (B,C,H,W)
        # record_len: (B)
        # pairwise_t_matrix: (B,L,L,4,4)
        # prior_encoding: (B,3)
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W)]
        split_x = self.regroup(x, record_len)
        # (B,L,L,2,3)
        pairwise_t_matrix = get_discretized_transformation_matrix(
            pairwise_t_matrix.reshape(-1, L, 4, 4), self.discrete_ratio,
            self.downsample_rate).reshape(B, L, L, 2, 3)
        # (B*L,L,1,H,W)
        roi_mask = get_rotated_roi((B * L, L, 1, H, W),
                                   pairwise_t_matrix.reshape(B * L * L, 2, 3))
        roi_mask = roi_mask.reshape(B, L, L, 1, H, W)

        batch_node_features = split_x
        # iteratively update the features for num_iteration times
        for l in range(self.num_iteration):

            batch_updated_node_features = []
            # iterate each batch
            for b in range(B):

                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                updated_node_features = []
                # update each node i
                for i in range(N):
                    # (N,1,H,W)
                    mask = roi_mask[b, :N, i, ...]

                    # flip the feature so the transformation is correct
                    batch_node_feature = batch_node_features[b]
                    batch_node_feature = rearrange(batch_node_feature,
                                                   'b c h w  -> b c w h')
                    batch_node_feature = torch.flip(batch_node_feature,
                                                    dims=(3,))


                    current_t_matrix = t_matrix[:, i, :, :]
                    current_t_matrix = get_transformation_matrix(
                        current_t_matrix, (H, W))

                    # (N,C,H,W)
                    neighbor_feature = warp_affine(batch_node_feature,
                                                   current_t_matrix,
                                                   (H, W))
                    # (N,C,H,W)
                    ego_agent_feature = batch_node_feature[i].unsqueeze(
                        0).repeat(N, 1, 1, 1)
                    # (N,1,H,W)
                    if self.use_mask:
                        AgentWeight = self.pixel_weighted_fusion(
                            torch.cat([neighbor_feature, ego_agent_feature],
                                  dim=1),
                            mask)
                    else:
                        AgentWeight = self.pixel_weighted_fusion(
                            torch.cat([neighbor_feature, ego_agent_feature],
                                      dim=1))

                    # (C,H,W)
                    ego_updated_features = (
                                AgentWeight * neighbor_feature * mask).sum(0)
                    ego_updated_features = torch.flip(ego_updated_features,
                                                      dims=(2, ))
                    ego_updated_features = rearrange(ego_updated_features,
                                                     'c w h -> c h w ')

                    updated_node_features.append(
                        ego_updated_features.unsqueeze(0))
                # (N,C,H,W)
                batch_updated_node_features.append(
                    torch.cat(updated_node_features, dim=0))
            batch_node_features = batch_updated_node_features

        # (B,C,H,W)
        out = torch.cat(
            [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        # (B,C,H,W)
        out = self.mlp(out.permute(0, 2, 3, 1))
        return out
