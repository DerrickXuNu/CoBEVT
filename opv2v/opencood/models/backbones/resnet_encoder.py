import torch
import torch.nn as nn
import torchvision.models as models

from einops import rearrange


class ResnetEncoder(nn.Module):
    """
    Resnet family to encode image.

    Parameters
    ----------
    params: dict
        The parameters of resnet encoder.
    """
    def __init__(self, params):
        super(ResnetEncoder, self).__init__()

        self.num_layers = params['num_layers']
        self.pretrained = params['pretrained']

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
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

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
        'num_layers': 34,
        'pretrained': True}

    model = ResnetEncoder(resnet_params)
    model.cuda()
    device = torch.device('cuda')

    for j, batch_data in enumerate(data_loader):
        cam_data = train_utils.to_device(batch_data['ego']['inputs'],
                                         device)
        output = model(cam_data)
        print('test passed')

