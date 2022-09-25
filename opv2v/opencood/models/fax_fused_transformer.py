"""
Implementation of Brady Zhou's cross view transformer
"""

import torch.nn as nn
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead


class FaxFusedTransformer(nn.Module):
    def __init__(self, config):
        super(FaxFusedTransformer, self).__init__()
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        cvm_params = config['fax']
        cvm_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(cvm_params)

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
        x = self.fax(batch_dict)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        output_dict = self.seg_head(x, b, l)

        return output_dict


if __name__ == '__main__':
    import os
    import torch
    from opencood.hypes_yaml.yaml_utils import load_yaml

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    test_data = torch.rand(1, 2, 4, 512, 512, 3)
    test_data = test_data.cuda()

    extrinsic = torch.rand(1, 2, 4, 4, 4)
    intrinsic = torch.rand(1, 2, 4, 3, 3)

    extrinsic = extrinsic.cuda()
    intrinsic = intrinsic.cuda()

    params = load_yaml('../hypes_yaml/opcamera/fax.yaml')

    model = FaxFusedTransformer(params['model']['args'])
    model = model.cuda()
    while True:
        output = model({'inputs': test_data,
                        'extrinsic': extrinsic,
                        'intrinsic': intrinsic})
        print('test_passed')
