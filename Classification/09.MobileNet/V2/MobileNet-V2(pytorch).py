import torch.nn as nn
from torchsummary import summary

class inverted_residual_block(nn.Module):
    def __init__(self,i,t,c,s,shortcut=False):
        super(inverted_residual_block,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(i,i*t,1,1),
            nn.BatchNorm2d(i*t),
            nn.ReLU6()
        )

        self.dconv = nn.Sequential(
            nn.Conv2d(i*t,i*t,3,s,1,groups=i*t),
            nn.BatchNorm2d(i*t),
            nn.ReLU6()
        )

        self.linearconv = nn.Sequential(
            nn.Conv2d(i*t,c,1,1),
            nn.BatchNorm2d(c)
        )

        self.shortcut = shortcut

    def forward(self,x):

        out = self.conv(x)
        out = self.dconv(out)
        out = self.linearconv(out)

        if self.shortcut :
            out += x

        return out



class mobilenetv2(nn.Module):
    def __init__(self,w=1.0):
        super(mobilenetv2,self).__init__()

        def make_layers(i,t,c,n,s):
            layers = []
            layers.append(inverted_residual_block(i,t,c,s))
            for i in range(1,n):
                layers.append(inverted_residual_block(c,t,c,1,True))
            return nn.Sequential(*layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,int(32*w),3,2,1),
            nn.BatchNorm2d(int(32*w)),
            nn.ReLU6()
        )

        self.irb1 = make_layers(int(32*w),1,int(16*w),1,1)
        self.irb2 = make_layers(int(16*w),6,int(24*w),2,2)
        self.irb3 = make_layers(int(24*w),6,int(32*w),3,2)
        self.irb4 = make_layers(int(32*w),6,int(64*w),4,2)
        self.irb5 = make_layers(int(64*w),6,int(96*w),3,1)
        self.irb6 = make_layers(int(96*w),6,int(160*w),3,2)
        self.irb7 = make_layers(int(160*w),6,int(320*w),1,1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(int(320*w),int(1280*w),1,1),
            nn.BatchNorm2d(int(1280*w)),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.conv3 = nn.Linear(int(1280*w),1000)

    def forward(self,x):

        out = self.conv1(x)
        out = self.irb1(out)
        out = self.irb2(out)
        out = self.irb3(out)
        out = self.irb4(out)
        out = self.irb5(out)
        out = self.irb6(out)
        out = self.irb7(out)

        out = self.conv2(out)

        out = out.view(out.size(0),-1)
 
        out = self.conv3(out)

        return out
    
summary(mobilenetv2(),(3,224,224))