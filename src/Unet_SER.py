import torch
import torch.nn as nn

import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        self.ca5 = ChannelAttention(out_c * 16)
        self.ca6 = ChannelAttention(out_c * 16)

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

        x5 = self.ca5(feats[4])
        r5 = x5.sum(dim=1, keepdim=True)

        x6 = self.ca6(feats[5])
        r6 = x6.sum(dim=1, keepdim=True)

        se1 = self.SemanticEnhance(r1, [r2, r3, r4, r5, r6])
        se2 = self.SemanticEnhance(r2, [r1, r3, r4, r5, r6])
        se3 = self.SemanticEnhance(r3, [r1, r2, r4, r5, r6])
        se4 = self.SemanticEnhance(r4, [r1, r2, r3, r5, r6])
        se5 = self.SemanticEnhance(r5, [r1, r2, r3, r4, r6])
        se6 = self.SemanticEnhance(r6, [r1, r2, r3, r5, r4])

        x1 = x1 * se1
        x2 = x2 * se2
        x3 = x3 * se3
        x4 = x4 * se4
        x5 = x5 * se5
        x6 = x6 * se6

        return x1, x2, x3, x4, x5, x6

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(out_channel),
                                  nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(out_channel)
                                  )

    def forward(self, input):
        out = self.conv(input)
        return out

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.encoder1 = DoubleConv(3, 32)
        self.encoder1_down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder2_down = nn.MaxPool2d(2, 2)
        self.encoder3 = DoubleConv(64, 128)
        self.encoder3_down = nn.MaxPool2d(2, 2)
        self.encoder4 = DoubleConv(128, 256)
        self.encoder4_down = nn.MaxPool2d(2, 2)
        self.encoder5 = DoubleConv(256, 512)
        self.encoder5_down = nn.MaxPool2d(2, 2)
        self.encoder6 = DoubleConv(512, 512)

        self.refine = Refiner(32)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2))
        self.decoder1_up = DoubleConv(1024, 512)

        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, stride=2))
        self.decoder2_up = DoubleConv(512, 256)

        self.decoder3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2))
        self.decoder3_up = DoubleConv(256, 128)

        self.decoder4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2))
        self.decoder4_up = DoubleConv(128, 64)

        self.decoder5 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2))
        self.decoder5_up = DoubleConv(64, 32)

        self.decoder_output = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        e1 = self.encoder1(x)
        e1_down = self.encoder1_down(e1)
        e2 = self.encoder2(e1_down)
        e2_down = self.encoder2_down(e2)
        e3 = self.encoder3(e2_down)
        e3_down = self.encoder3_down(e3)
        e4 = self.encoder4(e3_down)
        e4_down = self.encoder4_down(e4)
        e5 = self.encoder5(e4_down)
        e5_down = self.encoder5_down(e5)
        e6 = self.encoder6(e5_down)

        e1, e2, e3, e4, e5, e6 = self.refine((e1, e2, e3, e4, e5, e6))

        d1 = self.decoder1(e6)
        d1 = torch.cat((d1, e5), dim=1)
        d1 = self.decoder1_up(d1)

        d2 = self.decoder2(d1)
        d2 = torch.cat((d2, e4), dim=1)
        d2 = self.decoder2_up(d2)

        d3 = self.decoder3(d2)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3_up(d3)

        d4 = self.decoder4(d3)
        d4 = torch.cat((d4, e2), dim=1)
        d4 = self.decoder4_up(d4)

        d5 = self.decoder5(d4)
        d5 = torch.cat((d5, e1), dim=1)
        d5 = self.decoder5_up(d5)

        out = self.decoder_output(d5)
        # print(out.shape)

        return out
