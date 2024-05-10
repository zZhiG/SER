import torch
import torch.nn as nn

from norms import BatchChannelNorm


class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out

class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out

class DSCblock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = DSC(c_in, c_out, k_size, stride, padding)
        self.bn = BatchChannelNorm(c_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class IDSCblock(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super().__init__()

        self.conv = IDSC(c_in, c_out, k_size, stride, padding)
        self.bn = BatchChannelNorm(c_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, c_out, k_size=3, stride=1, padding=1, block=DSCblock, order=3):
        super().__init__()
        self.convs = nn.ModuleList([block(c_out, c_out, k_size, stride, padding) for _ in range(order)])
        self.res = nn.Conv2d(c_out*2, c_out, 1)
        self.order = order

    def forward(self, x):
        x_0 = x
        for i in range(self.order):
            x = self.convs[i](x)
        x = torch.cat([x, x_0], dim=1)
        return self.res(x)

class Encoder(nn.Module):
    def __init__(self, in_c=3, out_c=16, orders=[2, 2, 2, 2]):
        super().__init__()
        self.layer0 = nn.Sequential(DSCblock(in_c, out_c),
                                    DSCblock(out_c, out_c, 2, 2, 0))

        self.layer1 = ResidualBlock(out_c, order=orders[0])
        self.pool1 = DSCblock(out_c, out_c * 2, 2, 2, 0)

        self.layer2 = ResidualBlock(out_c * 2, order=orders[1])
        self.pool2 = DSCblock(out_c * 2, out_c * 4, 2, 2, 0)

        self.layer3 = ResidualBlock(out_c * 4, order=orders[2])
        self.pool3 = DSCblock(out_c * 4, out_c * 8, 2, 2, 0)

        self.layer4 = ResidualBlock(out_c * 8, order=orders[3])

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)

        x2 = self.pool1(x1)
        x2 = self.layer2(x2)

        x3 = self.pool2(x2)
        x3 = self.layer3(x3)

        x4 = self.pool3(x3)
        x4 = self.layer4(x4)

        return x1, x2, x3, x4
