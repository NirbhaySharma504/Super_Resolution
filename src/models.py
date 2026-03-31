"""
ESRGAN model architecture.
DenseBlock, RRDB, ESRGANGenerator, VGGStyleDiscriminator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.c1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.c2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.c3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.c4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.c5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
        return self.c5(torch.cat([x, x1, x2, x3, x4], 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.b1 = DenseBlock(nf, gc)
        self.b2 = DenseBlock(nf, gc)
        self.b3 = DenseBlock(nf, gc)

    def forward(self, x):
        return self.b3(self.b2(self.b1(x))) * 0.2 + x


class ESRGANGenerator(nn.Module):
    def __init__(self, ic=3, oc=3, nf=64, nr=8, gc=32, ts=125):
        super().__init__()
        self.ts = ts
        self.c0 = nn.Conv2d(ic, nf, 3, 1, 1)
        self.trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nr)])
        self.tc = nn.Conv2d(nf, nf, 3, 1, 1)
        self.u1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.u2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.hc = nn.Conv2d(nf, nf, 3, 1, 1)
        self.fc = nn.Conv2d(nf, oc, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        f = self.c0(x)
        f = f + self.tc(self.trunk(f))
        f = self.act(self.u1(F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)))
        f = self.act(self.u2(F.interpolate(f, size=self.ts, mode='bilinear', align_corners=False)))
        return self.fc(self.act(self.hc(f)))


class VGGStyleDiscriminator(nn.Module):
    def __init__(self, ic=3, nf=64):
        super().__init__()

        def db(i, o, s=1):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(i, o, 3, s, 1)),
                nn.LeakyReLU(0.2, True),
            )

        self.feat = nn.Sequential(
            db(ic, nf), db(nf, nf, 2),
            db(nf, nf * 2), db(nf * 2, nf * 2, 2),
            db(nf * 2, nf * 4), db(nf * 4, nf * 4, 2),
            db(nf * 4, nf * 8), db(nf * 8, nf * 8, 2),
        )
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nf * 8, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.cls(self.feat(x))
