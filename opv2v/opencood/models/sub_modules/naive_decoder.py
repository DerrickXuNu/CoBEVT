from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NaiveDecoder(nn.Module):
    """
    A Naive decoder implementation

    Parameters
    ----------
    params: dict

    Attributes
    ----------
    num_ch_dec : list
        The decoder layer channel numbers.

    num_layer : int
        The number of decoder layers.

    input_dim : int
        The channel number of the input to
    """
    def __init__(self, params):
        super(NaiveDecoder, self).__init__()

        self.num_ch_dec = params['num_ch_dec']
        self.num_layer = params['num_layer']
        self.input_dim = params['input_dim']

        assert len(self.num_ch_dec) == self.num_layer

        # decoder
        self.convs = OrderedDict()
        for i in range(self.num_layer-1, -1, -1):
            # upconv_0
            num_ch_in = self.input_dim if i == self.num_layer-1\
                else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 0)] = nn.Conv2d(
                num_ch_in, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] = nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(
                num_ch_out, num_ch_out, 3, 1, 1)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 1)] = nn.ReLU(True)
        self.decoder = nn.ModuleList(list(self.convs.values()))

    @staticmethod
    def upsample(x):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x):
        """
        Upsample to

        Parameters
        ----------
        x : torch.tensor
            The bev bottleneck feature, shape: (B, L, C1, H, W)

        Returns
        -------
        Output features with (B, L, C2, H, W)
        """
        b, l, c, h, w = x.shape
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        for i in range(self.num_layer-1, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)

            x = self.upsample(x)

            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)
            x = self.convs[("relu", i, 1)](x)

        x = rearrange(x, '(b l) c h w -> b l c h w',
                      b=b, l=l)
        return x
