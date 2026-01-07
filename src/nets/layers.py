
import torch
import torch.nn as nn
import torch.nn.functional as F
from .masker import Masker
from .carafe import CARAFE


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(True)
    )


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias), nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size, padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=4, mode="bilinear"),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear"),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1),
        )
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        """
        x: b, 512, 26, 26
        word: b, 512
        """
        x = self.vis(x)
        B, C, H, W = x.size()

        x = x.reshape(1, B * C, H, W)

        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        out = F.conv2d(x, weight, padding=self.kernel_size // 2, groups=weight.size(0), bias=bias)
        out = out.transpose(0, 1)
        return out


class FPN_AD(nn.Module):
    def __init__(self, in_channels=[512, 1024, 1024], out_channels=[256, 512, 1024]):
        super(FPN_AD, self).__init__()

        self.txt_proj = linear_layer(in_channels[2], out_channels[2])

        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]), nn.ReLU(True))

        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1], out_channels[1], 1, 0)

        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1], out_channels[1], 1, 0)

        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)

        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1), conv_layer(out_channels[1], out_channels[1], 3, 1)
        )
        self.masker3 = Masker(512, 512)
        self.masker4 = Masker(512, 512)
        self.masker5 = Masker(512, 512)
        self.carafe = CARAFE(in_channels[2], in_channels[2], up_factor=2)

    def forward(self, imgs, state):
        v3, v4, v5 = imgs

        fq3 = F.avg_pool2d(v3, 2, 2)

        mask3_sup = self.masker3(fq3)
        mask3_inf = torch.ones_like(mask3_sup) - mask3_sup
        fq3_sup = fq3 * mask3_sup
        fq3_inf = fq3 * mask3_inf

        f4 = self.f2_v_proj(v4)
        fq4 = self.f4_proj4(f4)

        mask4_sup = self.masker4(fq4)
        mask4_inf = torch.ones_like(mask4_sup) - mask4_sup
        fq4_sup = fq4 * mask4_sup
        fq4_inf = fq4 * mask4_inf

        fq5 = self.carafe(v5)

        mask5_sup = self.masker5(fq5)
        mask5_inf = torch.ones_like(mask5_sup) - mask5_sup
        fq5_sup = fq5 * mask5_sup
        fq5_inf = fq5 * mask5_inf

        fq_sup = torch.cat([fq3_sup, fq4_sup, fq5_sup], dim=1)
        fq_sup = self.aggr(fq_sup)

        fq_inf = torch.cat([fq3_inf, fq4_inf, fq5_inf], dim=1)
        fq_inf = self.aggr(fq_inf)
        return fq_sup, fq_inf
