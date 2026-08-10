import torch
from torch import nn
from torch.nn.utils import spectral_norm
#-----------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c,kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
#-----------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
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

    # -------- Encoder --------
        self.e1 = EncoderBlock(1, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 256)
        self.e5 = EncoderBlock(256, 512)
        self.e6 = EncoderBlock(512, 512)
        self.e7 = EncoderBlock(512, 512)

        # Bottleneck
        self.resblocks = nn.Sequential(
            *[ResBlock(512) for _ in range(6)]
        )

        # -------- Decoder --------
        self.d1 = DecoderBlock(512, 512)
        self.d2 = DecoderBlock(512 + 512, 512)
        self.d3 = DecoderBlock(512 + 512, 256)

        # Self-Attention
        self.attn = SelfAttention(256)

        self.d4 = DecoderBlock(256 + 256, 256)
        self.d5 = DecoderBlock(256 + 256, 128)
        self.d6 = DecoderBlock(128 + 128, 64)

        self.final = nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=5,stride=2,padding=2, output_padding=1 )

        self.tanh = nn.Tanh()
       
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        b = self.resblocks(e7)

        d1 = self.d1(b)
        d1 = torch.cat([d1, e6], dim=1)

        d2 = self.d2(d1)
        d2 = torch.cat([d2, e5], dim=1)

        d3 = self.d3(d2)
        d3 = self.attn(d3)          # Attention

        d3 = torch.cat([d3, e4], dim=1)

        d4 = self.d4(d3)
        d4 = torch.cat([d4, e3], dim=1)

        d5 = self.d5(d4)
        d5 = torch.cat([d5, e2], dim=1)

        d6 = self.d6(d5)
        d6 = torch.cat([d6, e1], dim=1)

        out = self.final(d6)

        return self.tanh(out)
#-----------------------------------------------------------------
    
# class Generator(nn.Module):

#     def __init__(self, in_channels=1, out_channels=1):
#         super().__init__()

#         # Encoder
#         self.e1 = EncoderBlock(1, 64, 7, 1, 3)
#         self.e2 = EncoderBlock(64, 128)
#         self.e3 = EncoderBlock(128, 256)

#         # ResNet blocks (6 blocks) 
#         self.resblocks = nn.Sequential(
#             *[ResBlock(256) for _ in range(6)]
#         )

#         # Decoder
#         self.d1 = DecoderBlock(256, 128)

#         # Attention
#         self.attn = SelfAttention(128)
        
#         self.d2 = DecoderBlock(128, 64)

#         self.final = nn.ConvTranspose2d(64 ,out_channels,7,1,3,0)
#         self.tanh = nn.Tanh()
       
#     def forward(self, x):
#         e1 = self.e1(x)
#         e2 = self.e2(e1)
#         e3 = self.e3(e2)
#         r1 = self.resblocks(e3)
#         d1 = self.d1(r1)
#         a1 = self.attn(d1)
#         d2 = self.d2(a1)
#         out = self.final(d2)
#         return self.tanh(out)
#-----------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        self.model = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 16x16 (stride=1, keep size)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 30x30 Patch output approximation
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, mri, pet):
        x = torch.cat([mri, pet], dim=1)  # concat along channels
        return self.model(x)

#-----------------------------------------------------------------
# class Discriminator(nn.Module):
#     def __init__(self, in_channels=2):
#         super().__init__()
#         # Conv1
#         self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
#         self.leaky1 = nn.LeakyReLU(0.2, inplace=True)

#         # Conv2
#         self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
#         self.leaky2 = nn.LeakyReLU(0.2, inplace=True)

#         # Self-Attention بعد از Conv2
#         self.attn = SelfAttention(128)

#         # Conv3
#         self.conv3 = spectral_norm(nn.Conv2d(128, 256, 4, 2, 1))
#         self.leaky3 = nn.LeakyReLU(0.2, inplace=True)

#         # Conv4
#         self.conv4 = spectral_norm(nn.Conv2d(256, 512, 4, 1, 1))
#         self.leaky4 = nn.LeakyReLU(0.2, inplace=True)

#         # Conv5
#         self.conv5 = spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))

#     def forward(self, mri, pet):
#         x = torch.cat([mri, pet], dim=1)
#         x = self.leaky1(self.conv1(x))
#         x = self.leaky2(self.conv2(x))
#         x = self.attn(x)  # Self-Attention روی feature map با 128 کانال
#         x = self.leaky3(self.conv3(x))
#         x = self.leaky4(self.conv4(x))
#         x = self.conv5(x)
#         return x