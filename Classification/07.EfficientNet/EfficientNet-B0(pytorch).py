import torch.nn as nn
from torchsummary import summary

class swish(nn.Module):
    def __init__(self):
        super(swish,self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = self.sigmoid(x)
        return x*out

class se_block(nn.Module):
    def __init__(self,c,r=4):
        super(se_block,self).__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)

        self.ex = nn.Sequential(
            nn.Linear(c,int(c/r)),
            nn.ReLU(),
            nn.Linear(int(c/r),c),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        sq = self.sq(x)
        sq = sq.view(sq.size(0),-1)

        ex = self.ex(sq)
        ex = ex.view(ex.size(0), ex.size(1), 1, 1)

        out = x*ex

        return out

class MBConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,e):
        super(MBConv,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*e,1,1),
            nn.BatchNorm2d(in_channels*e),
            swish()
        )

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels*e,in_channels*e,kernel_size,stride,int((kernel_size-1)/2),groups=in_channels*e),
            nn.BatchNorm2d(in_channels*e),
            swish()
        )

        self.se_block = nn.Sequential(
            se_block(in_channels*e),
            swish()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*e,out_channels,1,1),
            nn.BatchNorm2d(out_channels)
        )

        self.residual = stride==1 and in_channels == out_channels
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.dconv(out)
        out = self.se_block(out)
        out = self.conv2(out)

        if self.residual:
            out += x

        return out

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,3,2,1),
            nn.BatchNorm2d(32),
            swish()
        )

        self.mbconv1 = MBConv(32,16,3,1,1)

        self.mbconv2 = nn.Sequential(
            MBConv(16,24,3,1,6),
            MBConv(24,24,3,2,6)
        )

        self.mbconv3 = nn.Sequential(
            MBConv(24,40,5,1,6),
            MBConv(40,40,5,2,6)
        )

        self.mbconv4 = nn.Sequential(
            MBConv(40,80,3,1,6),
            MBConv(80,80,3,1,6),
            MBConv(80,80,3,2,6)
        )

        self.mbconv5 = nn.Sequential(
            MBConv(80,112,5,1,6),
            MBConv(112,112,5,1,6),
            MBConv(112,112,5,1,6)
        )

        self.mbconv6 = nn.Sequential(
            MBConv(112,192,5,1,6),
            MBConv(192,192,5,1,6),
            MBConv(192,192,5,1,6),
            MBConv(192,192,5,2,6)
        )

        self.mbconv7 = MBConv(192,320,3,1,6)

        self.conv2 = nn.Sequential(
            nn.Conv2d(320,1280,1,1),
            nn.BatchNorm2d(1280),
            swish(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.FC = nn.Sequential(
            nn.Linear(1280,1280),
            nn.Softmax(1)
        )


    def forward(self,x):
        out = self.conv1(x)
        out = self.mbconv1(out)
        out = self.mbconv2(out)
        out = self.mbconv3(out)
        out = self.mbconv4(out)
        out = self.mbconv5(out)
        out = self.mbconv6(out)
        out = self.mbconv7(out)
        out = self.conv2(out)
        out = out.view(out.size(0),-1)
        out = self.FC(out)

        return out

summary(EfficientNet(),(3,224,224))
