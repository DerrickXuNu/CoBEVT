"""
Implementation of Brady Zhou's cross view transformer
"""
import torch.nn as nn
from einops import rearrange
from opencood.models.sub_modules.cvt_modules import CrossViewModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.fusion_modules.v2v_fuse import V2VNetFusion
from opencood.models.sub_modules.bev_seg_head import BevSegHead


class CrossViewTransformerV2VNet(nn.Module):
    def __init__(self, config):
        super(CrossViewTransformerV2VNet, self).__init__()
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        cvm_params = config['cvm']
        cvm_params['backbone_output_shape'] = self.encoder.output_shapes
        self.cvm = CrossViewModule(cvm_params)

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']

        # spatial fusion
        self.fusion_net = V2VNetFusion(config['v2vnet_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])

    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        pairwise_t_matrix = batch_dict['pairwise_t_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.cvm(batch_dict)

        # B*L, C, H, W
        x = x.squeeze(1)
        # fuse all agents together to get a single bev map, b h w c
        x = self.fusion_net(x, record_len, pairwise_t_matrix, None)
        x = x.unsqueeze(1).permute(0, 1, 4, 2, 3)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        # L = 1 for sure in intermedaite fusion at this point
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        return output_dict
