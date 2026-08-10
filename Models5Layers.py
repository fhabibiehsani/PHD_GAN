import torch
from torch import nn
#-----------------------------------------------------------------
class Norm2d(nn.Module):
    def __init__(self,out_c):
        super().__init__()
        self.block = nn.Sequential(
            # nn.BatchNorm2d(out_c)
            nn.InstanceNorm2d(out_c, affine=True)
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
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, bn_momentum=0.8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn1 = Norm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat([x, out], dim=1)  # concatenate with input (dense connection)
        return out
#-----------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            Norm2d(dim),
            nn.ReLU(True),
           # nn.Dropout2d(0.5),
            nn.Conv2d(dim, dim, 3, 1, 1),
            Norm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)
#-----------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=5, stride=2, padding=2),
            Norm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
#-----------------------------------------------------------------
class EncoderDenseBlock(nn.Module):
    def __init__(self,in_c,out_c,num_dense_layers=2):
        super().__init__()
        layers=[]
        ch=in_c
        for _ in range(num_dense_layers):
            layers.append(DenseBlock( ch,growth_rate=out_c//num_dense_layers)  )
            ch += out_c//num_dense_layers
        self.dense_block=nn.Sequential(*layers)
        self.compress=nn.Conv2d(ch,out_c,kernel_size=1)
        self.downsample=nn.Conv2d( out_c, out_c,kernel_size=2,stride=2)

    def forward(self,x):
        out=self.dense_block(x)
        skip=self.compress(out)
        down=self.downsample(skip)
        return down, skip
#-----------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c,kernel_size=5,stride=2,padding=2,output_padding=1),
            Norm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
#-----------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, in_c=1, out_c=1):
        super().__init__()

        # -------- Encoder --------
        self.e1 = EncoderBlock(in_c, 64)              # 128x128
        self.e2 = EncoderBlock(64, 128)               # 64x64
        self.e3 = EncoderBlock(128, 256)              # 32x32
        self.e4 = EncoderBlock(256, 512)              # 16x16

        self.e5 = EncoderBlock(512, 512)              # 8x8 (bottleneck)

        # -------- Decoder --------
        self.d1 = DecoderBlock(512, 512)              # 16x16
        self.d2 = DecoderBlock(512 + 512, 256)        # 32x32
        self.d3 = DecoderBlock(256 + 256, 128)        # 64x64
        self.d4 = DecoderBlock(128 + 128, 64)         # 128x128

        # -------- Output --------
        self.final = nn.ConvTranspose2d(64 + 64, out_c, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        e5 = self.e5(e4)

        # Decoder + Skip connections
        d1 = self.d1(e5)
        d2 = self.d2(torch.cat([d1, e4], dim=1))
        d3 = self.d3(torch.cat([d2, e3], dim=1))
        d4 = self.d4(torch.cat([d3, e2], dim=1))

        # final skip with first encoder
        out = self.final(torch.cat([d4, e1], dim=1))
        return self.tanh(out)
#-----------------------------------------------------------------
class DenseUNet(nn.Module):
    def __init__(self, in_c=1, out_c=1):
        super().__init__()
        
        # -------- Encoder --------
        self.e1 = EncoderDenseBlock(in_c, 64)      # 128
        self.e2 = EncoderDenseBlock(64, 128)       # 64
        self.e3 = EncoderDenseBlock(128, 256)      # 32
        self.e4 = EncoderDenseBlock(256, 512)      # 16

        self.e5 = EncoderDenseBlock(512, 512)      # 8 (bottleneck)
        
        # -------- Decoder --------
        self.d1 = DecoderBlock(512, 512)
        self.d2 = DecoderBlock(512 + 512, 256)
        self.d3 = DecoderBlock(256 + 256, 128)
        self.d4 = DecoderBlock(128 + 128, 64)

        # -------- Output --------
        self.final = nn.ConvTranspose2d(64 + 64, out_c, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):

        # -------- Encoder --------
        e1, e1_skip = self.e1(x)
        e2, e2_skip = self.e2(e1)
        e3, e3_skip = self.e3(e2)
        e4, e4_skip = self.e4(e3)

        e5, e5_skip = self.e5(e4)

        # -------- Decoder + Skip connections --------
        d1 = self.d1(e5)
        d2 = self.d2( torch.cat([d1, e4_skip], dim=1))
        d3 = self.d3( torch.cat([d2, e3_skip], dim=1))
        d4 = self.d4( torch.cat([d3, e2_skip], dim=1))

        # -------- Output --------
        out = self.final( torch.cat([d4, e1_skip], dim=1))
        return self.tanh(out)
#-----------------------------------------------------------------
class ResUNet(nn.Module):
    def __init__(self, in_c=1, out_c=1):
        super().__init__()

        # -------- Encoder --------
        self.e1 = EncoderBlock(in_c, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        self.e5 = EncoderBlock(512, 512)  

        # -------- ResNet blocks --------
        self.resblocks = nn.Sequential(*[ResBlock(512) for _ in range(3)])

        # -------- Decoder --------
        self.d1 = DecoderBlock(512, 512)
        self.d2 = DecoderBlock(512 + 512, 256)
        self.d3 = DecoderBlock(256 + 256, 128)
        self.d4 = DecoderBlock(128 + 128, 64)

        # -------- Self-Attention --------
        self.attn = SelfAttention(128)

        # -------- Output --------
        self.final = nn.ConvTranspose2d(64 + 64, out_c, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):

        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        # ResNet bottleneck
        b = self.resblocks(e5)

        # Decoder
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e4], dim=1))
        d3 = self.d3(torch.cat([d2, e3], dim=1))
        # Residual Self-Attention
        d3 = d3 + self.attn(d3)
        d4 = self.d4(torch.cat([d3, e2], dim=1))
        out = self.final(torch.cat([d4, e1], dim=1))
        return self.tanh(out)
#-----------------------------------------------------------------
class DRUNet(nn.Module):
    def __init__(self, in_c=1, out_c=1):
        super().__init__()

        # -------- Encoder --------
        self.e1 = EncoderDenseBlock(in_c, 64)
        self.e2 = EncoderDenseBlock(64, 128)
        self.e3 = EncoderDenseBlock(128, 256)
        self.e4 = EncoderDenseBlock(256, 512)

        self.e5 = EncoderDenseBlock(512, 512)  

        # -------- ResNet blocks --------
        self.resblocks = nn.Sequential(*[ResBlock(512) for _ in range(3)])

        # -------- Decoder --------
        self.d1 = DecoderBlock(512, 512)
        self.d2 = DecoderBlock(512 + 512, 256)
        self.d3 = DecoderBlock(256 + 256, 128)
        self.d4 = DecoderBlock(128 + 128, 64)

        # -------- Self-Attention --------
        self.attn = SelfAttention(128)

        # -------- Output --------
        self.final = nn.ConvTranspose2d(64 + 64, out_c, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):

        # Encoder
        e1, e1_skip = self.e1(x)
        e2, e2_skip = self.e2(e1)
        e3, e3_skip = self.e3(e2)
        e4, e4_skip = self.e4(e3)

        e5, e5_skip = self.e5(e4)

        # ResNet bottleneck
        b = self.resblocks(e5)

        # -------- Decoder + Skip connections --------
        d1 = self.d1(e5)
        d2 = self.d2( torch.cat([d1, e4_skip], dim=1))
        d3 = self.d3( torch.cat([d2, e3_skip], dim=1))

          # Residual Self-Attention
        d3 = d3 + self.attn(d3)
        d4 = self.d4( torch.cat([d3, e2_skip], dim=1))

        # -------- Output --------
        out = self.final( torch.cat([d4, e1_skip], dim=1))
        return self.tanh(out)
#-----------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_c=2):
        super().__init__()

        self.model = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(in_c, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            Norm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            Norm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 16x16 (stride=1, keep size)
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            Norm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 30x30 Patch output approximation
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, mri, pet):
        x = torch.cat([mri, pet], dim=1)  # concat along channels
        return self.model(x)
