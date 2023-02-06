import torch
import torch.nn as nn
from torchsummary import summary

class UNet2(nn.Module):
    def __init__(self,DSV=True):
        super(UNet2,self).__init__()

        self.DSV = DSV
        def cbr(in_channels, out_channels, kernel_size = 3, stride = 1):
            layers = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,1),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(out_channels,out_channels,kernel_size,stride,1),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            return layers

        def h(y,x,kernel_size = 3, stride = 1):
            in_channels = (64*2**y)*x+(64*2**(y+1))
            out_channels = 64*2**y
            
            layers = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,1),
                nn.ReLU()

            )
            return layers
        
        self.enc0_0 = cbr(1,64)
        self.enc1_0 = cbr(64,128)
        self.enc2_0 = cbr(128,256)
        self.enc3_0 = cbr(256,512)
        self.enc4_0 = cbr(512,1024)

        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.block0_1 = h(0,1,3,1)
        self.block1_1 = h(1,1,3,1)
        self.block2_1 = h(2,1,3,1)
        self.block3_1 = h(3,1,3,1)

        self.block0_2 = h(0,2,3,1)
        self.block1_2 = h(1,2,3,1)
        self.block2_2 = h(2,2,3,1)
        
        self.block0_3 = h(0,3,3,1)
        self.block1_3 = h(1,3,3,1)

        self.block0_4 = h(0,4,3,1)
        
        self.result = nn.Sequential(
            nn.Conv2d(64,1,1,1,1),
            nn.Sigmoid()
        )
        
      
    def forward(self,x):
        
        enc0_0 = self.enc0_0(x)
        pool1 = self.pool(enc0_0)

        enc1_0 = self.enc1_0(pool1)
        pool2 = self.pool(enc1_0)

        enc2_0 = self.enc2_0(pool2)
        pool3 = self.pool(enc2_0)

        enc3_0 = self.enc3_0(pool3)
        pool4 = self.pool(enc3_0)

        enc4_0 = self.enc4_0(pool4)

        block0_1 = self.block0_1(torch.cat([enc0_0,self.up(enc1_0)],1))
        block1_1 = self.block1_1(torch.cat([enc1_0,self.up(enc2_0)],1))
        block2_1 = self.block2_1(torch.cat([enc2_0,self.up(enc3_0)],1))
        block3_1 = self.block3_1(torch.cat([enc3_0,self.up(enc4_0)],1))

        block0_2 = self.block0_2(torch.cat([enc0_0,block0_1,self.up(block1_1)],1))
        block1_2 = self.block1_2(torch.cat([enc1_0,block1_1,self.up(block2_1)],1))
        block2_2 = self.block2_2(torch.cat([enc2_0,block2_1,self.up(block3_1)],1))

        block0_3 = self.block0_3(torch.cat([enc0_0,block0_1,block0_2,self.up(block1_2)],1))
        block1_3 = self.block1_3(torch.cat([enc1_0,block1_1,block1_2,self.up(block2_2)],1))
        
        block0_4 = self.block0_4(torch.cat([enc0_0,block0_1,block0_2,block0_3,self.up(block1_3)],1))

        out = []
        if self.DSV:
            out.append(self.result(block0_1))
            out.append(self.result(block0_2))
            out.append(self.result(block0_3))
            out.append(self.result(block0_4))
        else:
            out.append(self.result(block0_4))

        return out 

print(summary(UNet2(),(1,400,400)))