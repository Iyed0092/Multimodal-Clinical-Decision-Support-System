import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.act = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch)

        self.se = SEBlock(out_ch)
        self.drop = nn.Dropout3d(p_dropout) if p_dropout > 0 else nn.Identity()

        self.res_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.res_conv(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        out = self.drop(out)
        return self.act(out + residual)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, n_classes=3, base=16, p_dropout=0.1):
        super().__init__()

        # encoder
        self.enc1 = ResidualSEBlock(in_channels, base)
        self.enc2 = ResidualSEBlock(base, base * 2)
        self.enc3 = ResidualSEBlock(base * 2, base * 4, p_dropout=p_dropout)
        self.enc4 = ResidualSEBlock(base * 4, base * 8, p_dropout=p_dropout)

        self.pool = nn.MaxPool3d(2)

        # bottleneck
        self.bottleneck = ResidualSEBlock(base * 8, base * 16, p_dropout=p_dropout * 2)

        # decoder - upsampling + attention + decoder blocks
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.att4 = AttentionGate(F_g=base * 16, F_l=base * 8, F_int=base * 4)
        self.dec4 = ResidualSEBlock(base * 16 + base * 8, base * 8)

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.att3 = AttentionGate(F_g=base * 8, F_l=base * 4, F_int=base * 2)
        self.dec3 = ResidualSEBlock(base * 8 + base * 4, base * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.att2 = AttentionGate(F_g=base * 4, F_l=base * 2, F_int=base)
        self.dec2 = ResidualSEBlock(base * 4 + base * 2, base * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.att1 = AttentionGate(F_g=base * 2, F_l=base, F_int=max(base // 2, 1))
        self.dec1 = ResidualSEBlock(base * 2 + base, base)

        self.out_conv = nn.Conv3d(base, n_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.01)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.constant_(m.weight, 1)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
