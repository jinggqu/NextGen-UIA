import torch
import torch.nn as nn
import torch.nn.functional as F


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


def gumbel_softmax(logits, tau=1e-5):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = (logits + gumbel_noise) / tau
    return F.softmax(y, dim=-1)


class Masker(nn.Module):
    def __init__(self, in_dim=512, outdim=512):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.outdim = outdim
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=outdim, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.coordconv = CoordConv(in_dim, in_dim, 3, 1)
        self.coordconv2 = CoordConv(in_dim, in_dim, 3, 1)
        self.conv2 = conv_layer(in_dim, outdim, 3, 1, 1)

    def forward(self, x):
        x = self.coordconv(x)
        x = self.conv2(x)
        x = self.coordconv2(x)
        x = self.conv(x)
        x = self.bn(x)

        binary_feature_map = self.sig(x)

        return binary_feature_map
