import torch
import torch.nn as nn
from torchsummary import summary

class UNet3(nn.Module):
    def __init__(self,DSV=True):
        super(UNet3,self).__init__()

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
        
        self.h = nn.Sequential(
            nn.Conv2d(320,320,3,1,1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            
            nn.Conv2d(320,320,3,1,1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )

        
        self.enc0 = cbr(1,64)
        self.enc1 = cbr(64,128)
        self.enc2 = cbr(128,256)
        self.enc3 = cbr(256,512)
        self.enc4 = cbr(512,1024)

        self.pool = nn.MaxPool2d(2,2)
        
        self.enc0_dec0 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1)
        )
        self.enc0_dec1 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,64,3,1,1)
        )
        self.enc0_dec2 = nn.Sequential(
            nn.MaxPool2d(4,4),
            nn.Conv2d(64,64,3,1,1)
        )
        self.enc0_dec3 = nn.Sequential(
            nn.MaxPool2d(8,8),
            nn.Conv2d(64,64,3,1,1)
        )

        self.enc1_dec1 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1)
        )
        self.enc1_dec2 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,64,3,1,1)
        )
        self.enc1_dec3 = nn.Sequential(
            nn.MaxPool2d(4,4),
            nn.Conv2d(128,64,3,1,1)
        )

        self.enc2_dec2 = nn.Sequential(
            nn.Conv2d(256,64,3,1,1)
        )
        self.enc2_dec3 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,64,3,1,1)
        )

        self.enc3_dec3 = nn.Sequential(
            nn.Conv2d(512,64,3,1,1)
        )


        self.dec_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(320,64,3,1,1)
        )

        self.dec_up4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(320,64,3,1,1)
        )

        self.dec_up8 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(320,64,3,1,1)
        )

        self.enc4_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1024,64,3,1,1)
        )

        self.enc4_up4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(1024,64,3,1,1)
        )

        self.enc4_up8 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(1024,64,3,1,1)
        )

        self.enc4_up16 = nn.Sequential(
            nn.Upsample(scale_factor=16, mode='bilinear'),
            nn.Conv2d(1024,64,3,1,1)
        )
        
        self.result1 = nn.Sequential(
            nn.Conv2d(320,1,3,1,1),
            nn.Sigmoid()
        )
        self.result2 = nn.Sequential(
            nn.Conv2d(1024,1,3,1,1),
            nn.Sigmoid()
        )


        
      
    def forward(self,x):
        
        enc0 = self.enc0(x)
        pool1 = self.pool(enc0)

        enc1 = self.enc1(pool1)
        pool2 = self.pool(enc1)

        enc2 = self.enc2(pool2)
        pool3 = self.pool(enc2)

        enc3 = self.enc3(pool3)
        pool4 = self.pool(enc3)

        enc4 = self.enc4(pool4)

        dec3 = self.h(torch.cat([self.enc0_dec3(enc0),self.enc1_dec3(enc1),self.enc2_dec3(enc2),self.enc3_dec3(enc3),self.enc4_up2(enc4)],1))
        dec2 = self.h(torch.cat([self.enc0_dec2(enc0),self.enc1_dec2(enc1),self.enc2_dec2(enc2),self.dec_up2(dec3),self.enc4_up4(enc4)],1))
        dec1 = self.h(torch.cat([self.enc0_dec1(enc0),self.enc1_dec1(enc1),self.dec_up2(dec2),self.dec_up4(dec3),self.enc4_up8(enc4)],1))
        dec0 = self.h(torch.cat([self.enc0_dec0(enc0),self.dec_up2(dec1),self.dec_up4(dec2),self.dec_up8(dec3),self.enc4_up16(enc4)],1))

        out = []
        if self.DSV:
            out.append(self.result1(dec0))
            out.append(self.result1(dec1))
            out.append(self.result1(dec2))
            out.append(self.result1(dec3))
            out.append(self.result2(enc4))
        else:
            out.append(self.result1(dec0))

        return out 

print(summary(UNet3(),(1,400,400)))