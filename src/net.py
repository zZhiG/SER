from encoder import *

import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Sequential(nn.Conv2d(in_c, in_c, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        w = self.avg(x)
        w = self.linear(w)
        x = x * w

        return x

class Refiner(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        self.ca1 = ChannelAttention(out_c)
        self.ca2 = ChannelAttention(out_c * 2)
        self.ca3 = ChannelAttention(out_c * 4)
        self.ca4 = ChannelAttention(out_c * 8)

    def SemanticEnhance(self, r, relations):
        rshape = r.shape[-1]
        for f in relations:
            if f.shape[-1] > rshape:
                r = r * F.adaptive_avg_pool2d(f, (rshape, rshape))
            elif f.shape[-1] < rshape:
                r = r * F.interpolate(f, size=(rshape, rshape), mode='bilinear', align_corners=True)

        return r

    def forward(self, feats):
        x1 = self.ca1(feats[0])
        r1 = x1.sum(dim=1, keepdim=True)

        x2 = self.ca2(feats[1])
        r2 = x2.sum(dim=1, keepdim=True)

        x3 = self.ca3(feats[2])
        r3 = x3.sum(dim=1, keepdim=True)

        x4 = self.ca4(feats[3])
        r4 = x4.sum(dim=1, keepdim=True)

        se1 = self.SemanticEnhance(r1, [r2, r3, r4])
        se2 = self.SemanticEnhance(r2, [r1, r3, r4])
        se3 = self.SemanticEnhance(r3, [r1, r2, r4])
        se4 = self.SemanticEnhance(r4, [r1, r2, r3])

        x1 = x1 * se1
        x2 = x2 * se2
        x3 = x3 * se3
        x4 = x4 * se4

        return x1, x2, x3, x4


class Net(nn.Module):
    def __init__(self, in_c=3, out_c=16, orders=[4, 4, 4, 4]):
        super().__init__()
        self.encoder = Encoder(in_c, out_c, orders)

        self.refiner = Refiner(out_c)

        self.up = nn.PixelShuffle(2)

        self.d3 = nn.Sequential(IDSCblock((out_c * 2) + (out_c * 4), out_c * 4),
                                ResidualBlock(out_c * 4, order=orders[2]))

        self.d2 = nn.Sequential(IDSCblock(out_c * 3, out_c * 2),
                                ResidualBlock(out_c * 2, order=orders[1]))

        self.d1 = nn.Sequential(IDSCblock(out_c * 2 // 4 + out_c, out_c),
                                ResidualBlock(out_c, order=orders[0]))

        self.seg = nn.Sequential(self.up,
                                 IDSC(out_c // 4, 1))

    def forward(self, x):
        feats = self.encoder(x)

        feats = self.refiner(feats)

        x4 = self.up(feats[3])

        x3 = torch.cat([x4, feats[2]], dim=1)
        x3 = self.d3(x3)

        x2 = self.up(x3)
        x2 = torch.cat([x2, feats[1]], dim=1)
        x2 = self.d2(x2)

        x1 = self.up(x2)
        x1 = torch.cat([x1, feats[0]], dim=1)
        x1 = self.d1(x1)

        y = self.seg(x1)

        return y
