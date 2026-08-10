import torch
from torch import nn
#-----------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, num_dense_layers=2, bn_momentum=0.8):
        super().__init__()

        layers = []
        ch = in_c
        growth = out_c // num_dense_layers

        for _ in range(num_dense_layers):
            layers.append(DenseBlock(ch, growth_rate=growth, bn_momentum=bn_momentum))
            ch = ch + growth

        self.dense_block = nn.Sequential(*layers)
        self.out_c = ch  

        self.downsample = nn.Conv2d(ch, out_c, 2, 2)

    def forward(self, x):
        out = self.dense_block(x)
        return self.downsample(out)#, out
#-----------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
#-----------------------------------------------------------------
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
#-----------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder blocks
        self.e1 = EncoderBlock(in_channels, 64)  # 128 
        self.e2 = EncoderBlock(64 ,128)  # 64
        self.e3 = EncoderBlock(128,256) # 32
        self.e4 = EncoderBlock(256,256) # 16
        self.e5 = EncoderBlock(256,512) # 8
        self.e6 = EncoderBlock(512,512) # 4
        self.e7 = EncoderBlock(512,512) # 2
        
        # Decoder blocks
        self.d1 = DecoderBlock(512 , 512)
        self.d2 = DecoderBlock(512 + 512, 512)
        self.d3 = DecoderBlock(512 + 512, 256)
        self.d4 = DecoderBlock(256 + 256, 256)
        self.d5 = DecoderBlock(256 + 256, 128)
        self.d6 = DecoderBlock(128 + 128, 64)

        # Output conv
        self.final = nn.ConvTranspose2d(64 + 64,1,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.tanh = nn.Tanh()  # Pix2Pix output [-1,1]
    
    def forward(self, x):
        
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        
        # Decoder
        d1 = self.d1(e7)
        d2 = self.d2(torch.cat([d1, e6], dim=1))   # 512 + 512
        d3 = self.d3(torch.cat([d2, e5], dim=1))   # 512 + 512
        d4 = self.d4(torch.cat([d3, e4], dim=1))   # 512 + 512
        d5 = self.d5(torch.cat([d4, e3], dim=1))   # 256 + 256
        d6 = self.d6(torch.cat([d5, e2], dim=1))   # 128 + 128
        # final output (NO extra concat)
        out = self.final(torch.cat([d6, e1], dim=1))  #64+64
        return self.tanh(out)
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
