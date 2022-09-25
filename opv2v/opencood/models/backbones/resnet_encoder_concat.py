from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F

from einops import rearrange


class ResnetEncoderConcat(nn.Module):
    """
    Resnet family to encode image (multi-scale image concat).

    Parameters
    ----------
    params: dict
        The parameters of resnet encoder.   `
    """
    def __init__(self, params):
        super(ResnetEncoderConcat, self).__init__()

        self.num_layers = params['num_layers']
        self.pretrained = params['pretrained']
        self.fpn = params['fpn']
        self.conv2d = nn.Conv2d(params['conv_input_dim'],
                                params['conv_output_dim'],
                                kernel_size=1)

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if self.num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet "
                "layers".format(self.num_layers))

        self.encoder = resnets[self.num_layers](self.pretrained)
        if self.fpn is not None:
            layer1_dim = self.fpn['layer1']
            layer2_dim = self.fpn['layer2']
            layer3_dim = self.fpn['layer3']
            output_dim = self.fpn['output_dim']

            self.fpn_network = \
                torchvision.ops.FeaturePyramidNetwork([layer1_dim,
                                                       layer2_dim,
                                                       layer3_dim], output_dim)

    @staticmethod
    def upsample(x, n):
        """Upsample input tensor by a factor of n
        """
        return F.interpolate(x, scale_factor=n, mode="nearest")

    def forward(self, input_images):
        """
        Compute deep features from input images.
        todo: multi-scale feature support

        Parameters
        ----------
        input_images : torch.Tensor
            The input images have shape of (B,L,M,H,W,3), where L, M are
            the num of agents and num of cameras per agents.

        Returns
        -------
        features: torch.Tensor
            The deep features for each image with a shape of (B,L,M,C,H,W)
        """
        b, l, m, h, w, c = input_images.shape
        input_images = input_images.view(b*l*m, h, w, c)
        # b, h, w, c -> b, c, h, w
        input_images = input_images.permute(0, 3, 1, 2).contiguous()

        x = self.encoder.conv1(input_images)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        x = self.encoder.layer1(self.encoder.maxpool(x))
        x1 = self.encoder.layer2(x)
        x2 = self.encoder.layer3(x1)
        x3 = self.encoder.layer4(x2)

        if self.fpn is not None:
            fpn_input = OrderedDict()
            fpn_input['feat0'] = x1
            fpn_input['feat1'] = x2
            fpn_input['feat2'] = x3

            output = self.fpn_network(fpn_input)
            x1 = output['feat0']
            x2 = output['feat1']
            x3 = output['feat2']

        # upsample
        x3 = self.upsample(x3, 4)
        x2 = self.upsample(x2, 2)
        # concat
        x = torch.concat([x1, x2, x3], dim=1)
        # conv 2d to reduce dimension
        x = self.conv2d(x)

        x = rearrange(x, '(b l m) c h w -> b l m c h w',
                      b=b, l=l, m=m)

        return x


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from torch.utils.data import DataLoader
    from opencood.tools import train_utils

    params = load_yaml('/opencood/hypes_yaml/opcamera/base_camera.yaml')

    opencood_train_dataset = build_dataset(params, visualize=False, train=True)
    data_loader = DataLoader(opencood_train_dataset,
                             batch_size=4,
                             num_workers=8,
                             collate_fn=opencood_train_dataset.collate_batch,
                             shuffle=False,
                             pin_memory=False)

    resnet_params = {
        'conv_input_dim': 896,
        'conv_output_dim': 512,
        'num_layers': 34,
        'pretrained': True}

    model = ResnetEncoderConcat(resnet_params)
    model.cuda()
    device = torch.device('cuda')

    for j, batch_data in enumerate(data_loader):
        cam_data = train_utils.to_device(batch_data['ego']['inputs'],
                                         device)
        output = model(cam_data)
        print('test passed')

