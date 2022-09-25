"""
Implementation of Brady Zhou's cross view transformer
"""

import torch
import torch.nn as nn
from einops import rearrange
from opencood.models.sub_modules.cvt_modules import CrossViewModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead


class CrossViewTransformer(nn.Module):
    def __init__(self, config):
        super(CrossViewTransformer, self).__init__()
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        cvm_params = config['cvm']
        cvm_params['backbone_output_shape'] = self.encoder.output_shapes
        self.cvm = CrossViewModule(cvm_params)

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

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.cvm(batch_dict)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        output_dict = self.seg_head(x, b, l)

        return output_dict