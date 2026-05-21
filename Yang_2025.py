import torch
from torch import nn
from torch.nn.utils import spectral_norm
#-----------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)    #(in_dim=128)/8=16
        self.key   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim // 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H*W)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)

        attn = torch.softmax(torch.bmm(q.permute(0,2,1), k), dim=-1)
        out = torch.bmm(v, attn.permute(0,2,1)).view(B, C, H, W)

        return self.gamma * out + x
#-----------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)
#-----------------------------------------------------------------
class Generator(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
       
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # ResNet blocks (6 blocks)
        self.resblocks = nn.Sequential(
            *[ResBlock(256) for _ in range(6)]
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        # Attention
        self.attn = SelfAttention(128)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(64, out_channels, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.resblocks(x)
        x = self.dec1(x)
        x = self.attn(x)
        x = self.dec2(x)
        x = self.dec3(x)

        return x
#-----------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        # Conv1
        self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.leaky1 = nn.LeakyReLU(0.2, inplace=True)

        # Conv2
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        self.leaky2 = nn.LeakyReLU(0.2, inplace=True)

        # Self-Attention بعد از Conv2
        self.attn = SelfAttention(128)

        # Conv3
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
        self.leaky3 = nn.LeakyReLU(0.2, inplace=True)

        # Conv4
        self.conv4 = spectral_norm(nn.Conv2d(256, 512, 4, 1, 1))
        self.leaky4 = nn.LeakyReLU(0.2, inplace=True)

        # Conv5
        self.conv5 = spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))

    def forward(self, mri, pet):
        x = torch.cat([mri, pet], dim=1)
        x = self.leaky1(self.conv1(x))
        x = self.leaky2(self.conv2(x))
        x = self.attn(x)  # Self-Attention روی feature map با 128 کانال
        x = self.leaky3(self.conv3(x))
        x = self.leaky4(self.conv4(x))
        x = self.conv5(x)
        return x