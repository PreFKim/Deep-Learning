import torch
import torch.nn as nn
from torchsummary import summary

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        def cbr(in_channels, out_channels, kernel_size = 3, stride = 1):
            layers = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            return layers
        
        self.enc1_1 = cbr(1,64)
        self.enc1_2 = cbr(64,64)
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.enc2_1 = cbr(64,128)
        self.enc2_2 = cbr(128,128)
        self.pool2 = nn.MaxPool2d(2,2)

        self.enc3_1 = cbr(128,256)
        self.enc3_2 = cbr(256,256)
        self.pool3 = nn.MaxPool2d(2,2)

        self.enc4_1 = cbr(256,512)
        self.enc4_2 = cbr(512,512)
        self.pool4 = nn.MaxPool2d(2,2)

        self.enc5_1 = cbr(512,1024)
        self.enc5_2 = cbr(1024,1024)

        self.unpool4 = nn.ConvTranspose2d(1024,512,2,2)
        self.dec4_2 = cbr(1024,512)
        self.dec4_1 = cbr(512,512)

        self.unpool3 = nn.ConvTranspose2d(512,256,2,2)
        self.dec3_2 = cbr(512,256)
        self.dec3_1 = cbr(256,256)

        self.unpool2 = nn.ConvTranspose2d(256,128,2,2)
        self.dec2_2 = cbr(256,128)
        self.dec2_1 = cbr(128,128)

        self.unpool1 = nn.ConvTranspose2d(128,64,2,2)
        self.dec1_2 = cbr(128,64)
        self.dec1_1 = cbr(64,64)

        self.result = nn.Sequential(
            nn.Conv2d(64,1,3,1,1),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        
        enc1_1 = self.enc1_1(x) 
        enc1_2 = self.enc1_2(enc1_1) 
        pool1 = self.pool1(enc1_2) 
        
        enc2_1 = self.enc2_1(pool1) 
        enc2_2 = self.enc2_2(enc2_1) 
        pool2 = self.pool2(enc2_2) 
        
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1) 
        pool3 = self.pool3(enc3_2)        
        
        enc4_1 = self.enc4_1(pool3) 
        enc4_2 = self.enc4_2(enc4_1) 
        pool4 = self.pool4(enc4_2) 
        
        enc5_1 = self.enc5_1(pool4) 
        enc5_2 = self.enc5_2(enc5_1) 
        
        unpool4 = self.unpool4(enc5_2)
        dec4_2 = self.dec4_2(torch.cat((unpool4,enc4_2),1))
        dec4_1 = self.dec4_1(dec4_2) 
        
        unpool3 = self.unpool3(dec4_1) 
        dec3_2 = self.dec3_2(torch.cat((unpool3,enc3_2),1)) 
        dec3_1 = self.dec3_1(dec3_2) 
        
        unpool2 = self.unpool2(dec3_1) 
        dec2_2 = self.dec2_2(torch.cat((unpool2,enc2_2),1)) 
        dec2_1 = self.dec2_1(dec2_2) 
        
        unpool1 = self.unpool1(dec2_1) 
        dec1_2 = self.dec1_2(torch.cat((unpool1,enc1_2),1))
        dec1_1 = self.dec1_1(dec1_2) 

        out = self.result(dec1_1)
        return out 

print(summary(UNet(),(1,400,400)))