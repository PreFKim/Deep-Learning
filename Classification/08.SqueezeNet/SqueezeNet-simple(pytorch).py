import torch
import torch.nn as nn
from torchsummary import summary

class fire_module(nn.Module):
    def __init__(self,in_channels,s1,e1,e3):
        super(fire_module,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,s1,1,1),
            nn.BatchNorm2d(s1),
            nn.ReLU()
        )

        self.e1 = nn.Sequential(
            nn.Conv2d(s1,e1,1,1),
            nn.BatchNorm2d(e1),
            nn.ReLU()
        )

        self.e2 = nn.Sequential(
            nn.Conv2d(s1,e3,3,1,1),
            nn.BatchNorm2d(e3),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.conv1(x)
        e1 = self.e1(x)
        e2 = self.e2(x)
        out = torch.cat([e1,e2],1)
        return out

class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,96,7,2,3),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(3,2,1)
        
        self.fire2 = fire_module(96,16,64,64)
        self.fire3 = fire_module(128,16,64,64)
        self.fire4 = fire_module(128,32,128,128)
        self.fire5 = fire_module(256,32,128,128)
        self.fire6 = fire_module(256,48,192,192)
        self.fire7 = fire_module(384,48,192,192)
        self.fire8 = fire_module(384,64,256,256)
        self.fire9 = fire_module(512,64,256,256)

        self.dropout = nn.Dropout(0.5)

        self.FC = nn.Sequential(
            nn.Conv2d(512,1000,1,1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self,x):
        conv1 = self.conv1(x)
        maxpool = self.maxpool(conv1)
        fire2 = self.fire2(maxpool)
        fire3 = self.fire3(fire2)
        fire4 = self.fire4(fire2+fire3)
        maxpool = self.maxpool(fire4)
        fire5 = self.fire5(maxpool)
        fire6 = self.fire6(maxpool+fire5)
        fire7 = self.fire7(fire6)
        fire8 = self.fire8(fire6+fire7)
        maxpool = self.maxpool(fire8)
        fire9 = self.fire9(maxpool)
        dropout = self.dropout(fire9)
        
        output =self.FC(dropout)

        return nn.Softmax(output)

print(summary(SqueezeNet(),(3,224,224)))
