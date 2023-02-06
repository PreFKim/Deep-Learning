import torch.nn as nn
from torchsummary import summary

class depthwise_separable_conv(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(depthwise_separable_conv,self).__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,stride,1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        out = self.dconv(x)
        out = self.conv(out)

        return out

class MobileNet(nn.Module):
    def __init__(self,a=1):
        super(MobileNet,self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32*a,3,2,1),
            nn.BatchNorm2d(32*a),
            nn.ReLU()
        )

        self.Mobile = nn.Sequential(
            depthwise_separable_conv(32*a,64,1),
            depthwise_separable_conv(64,128,2),
            depthwise_separable_conv(128,128,1),
            depthwise_separable_conv(128,256,2),
            depthwise_separable_conv(256,256,1),
            depthwise_separable_conv(256,512,2),

            depthwise_separable_conv(512,512,1),
            depthwise_separable_conv(512,512,1),
            depthwise_separable_conv(512,512,1),
            depthwise_separable_conv(512,512,1),
            depthwise_separable_conv(512,512,1),

            depthwise_separable_conv(512,1024,1),
            depthwise_separable_conv(1024,1024,1),
            nn.AdaptiveAvgPool2d(1)

        )

        self.FC = nn.Sequential(
            nn.Linear(1024,1000),
            nn.Softmax()
        )

    def forward(self,x):
        out = self.conv1(x)
        out = self.Mobile(out)

        out = out.view(out.size(0),-1)

        out = self.FC(out)

        return out

print(summary(MobileNet(),(3,224,224)))

