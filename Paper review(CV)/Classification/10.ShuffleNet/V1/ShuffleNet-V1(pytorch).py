import torch
import torch.nn as nn
from torchsummary import summary


def channel_shuffle(input,group):
    b,c,h,w = input.size()
    ranges = c // group
    out = input.view(b,group,ranges,h,w)
    out = out.permute(0,2,1,3,4)
    out = out.reshape(b,c,h,w)
    return out

class shufflenet_unit(nn.Module):
    def __init__(self,in_channels,out_channels,stride,group,i=1):
        super(shufflenet_unit,self).__init__()

        self.g = group
        self.s = stride

        bottleneck_channels = out_channels//4

        if i == 0:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels,bottleneck_channels,1,1,groups=1),
                nn.BatchNorm2d(bottleneck_channels)
            )
        else :
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels,bottleneck_channels,1,1,groups=group),
                nn.BatchNorm2d(bottleneck_channels)
            )
        
        self.relu = nn.ReLU()

        if stride == 1:
            self.shortcut = nn.Sequential()
            final_channels = out_channels
        else :
            self.shortcut = nn.AvgPool2d(3,2,1)
            final_channels = out_channels-in_channels
        
        self.final = nn.Sequential(
            nn.Conv2d(bottleneck_channels,bottleneck_channels,3,stride,1,groups=bottleneck_channels),
            nn.BatchNorm2d(bottleneck_channels),
            nn.Conv2d(bottleneck_channels,final_channels,1,1,groups=group),
            nn.BatchNorm2d(final_channels)
        )


    def forward(self,x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = channel_shuffle(out,self.g)
        out = self.final(out)
        if self.s == 1:
            out += shortcut
        else:
            out = torch.cat([shortcut,out],1)

        out = self.relu(out)

        return out

class ShuffleNet(nn.Module):
    def __init__(self,g=1,s=1):
        super(ShuffleNet,self).__init__()

        repeat = [3,7,3]
        channel_list = {1:144,2:200,3:240,4:272,8:384}
        channels = channel_list[g]*s

        def make_layers(in_channels,out_channels,stage):
            layers = []
            layers.append(shufflenet_unit(in_channels,out_channels,2,g,stage))
            for _ in range(repeat[stage]):
                layers.append(shufflenet_unit(out_channels,out_channels,1,g,1))

            return nn.Sequential(*layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,24,3,2,1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )

        self.stage1 = make_layers(24,channels,0)
        self.stage2 = make_layers(channels,channels*2,1)
        self.stage3 = nn.Sequential(
            make_layers(channels*2,channels*4,2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.FC = nn.Sequential(
            nn.Linear(channels*4,1000)
        )

    def forward(self,x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        out = out.view(out.size(0),-1)
        out = self.FC(out)
        return out

summary(ShuffleNet(),(3,224,224))