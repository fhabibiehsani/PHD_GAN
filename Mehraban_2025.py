import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(111)  # Random Generator Seed
# ------------------------------
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, bn_momentum=0.8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(growth_rate, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat([x, out], dim=1)  # concatenate with input (dense connection)
        return out
# ------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_dense_layers=2, bn_momentum=0.8):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(num_dense_layers):
            layers.append(DenseBlock(ch, growth_rate=out_channels // num_dense_layers, bn_momentum=bn_momentum))
            ch += out_channels // num_dense_layers  # channel increases after each dense layer
        self.dense_block = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(ch, out_channels, kernel_size=2, stride=2)  # downsample by 2
    
    def forward(self, x):
        out = self.dense_block(x)
        out_down = self.downsample(out)
        return out_down, out  # return downsampled + skip features
# ------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bn_momentum=0.8):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, skip):
        x = self.relu(self.bn(self.upconv(x)))
        # اضافه کردن این خط برای مطابقت اندازه spatial قبل از concat
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv(x))
        return x
# ------------------------------
class DenseUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        # Encoder blocks
        self.enc1 = EncoderBlock(in_channels, base_channels)       # 256 -> 128
        self.enc2 = EncoderBlock(base_channels, base_channels*2)   # 128 -> 64
        self.enc3 = EncoderBlock(base_channels*2, base_channels*4) # 64 -> 32
        self.enc4 = EncoderBlock(base_channels*4, base_channels*8) # 32 -> 16
        self.enc5 = EncoderBlock(base_channels*8, base_channels*8) # bottleneck
        # Decoder blocks
        self.dec4 = DecoderBlock(base_channels*8, base_channels*8, base_channels*4)
        self.dec3 = DecoderBlock(base_channels*4, base_channels*4, base_channels*2)
        self.dec2 = DecoderBlock(base_channels*2, base_channels*2, base_channels)
        self.dec1 = DecoderBlock(base_channels, base_channels, base_channels)
        # Output conv
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()  # Pix2Pix output [-1,1]
    
    def forward(self, x):
        # Encoder
        x1_down, x1_skip = self.enc1(x)
        x2_down, x2_skip = self.enc2(x1_down)
        x3_down, x3_skip = self.enc3(x2_down)
        x4_down, x4_skip = self.enc4(x3_down)
        x5_down, _ = self.enc5(x4_down)  # bottleneck, no skip
        # Decoder
        d4 = self.dec4(x5_down, x4_skip)
        d3 = self.dec3(d4, x3_skip)
        d2 = self.dec2(d3, x2_skip)
        d1 = self.dec1(d2, x1_skip)
        out = self.tanh(self.final_conv(d1))
        return out
