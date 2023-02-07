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
    def __init__(self,in_channels,out_channels,stride,ratio=0.5):
        super(shufflenet_unit,self).__init__()

        if stride == 1:
            in_c = int(in_channels*ratio)
            out_c = int(out_channels*ratio)
        else :
            in_c = in_channels
            out_c = out_channels//2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c,in_c,3,2,1,groups=in_c),
                nn.BatchNorm2d(in_c),

                nn.Conv2d(in_c,out_c,1,1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.s = stride
        self.split = in_c
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,1,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.Conv2d(out_c,out_c,3,stride,1,groups=out_c),
            nn.BatchNorm2d(out_c),

            nn.Conv2d(out_c,out_c,1,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )


    def forward(self,x):
        if self.s == 1:
            out = self.conv(x[:,:self.split,:,:])
            shortcut = x[:,self.split:,:,:]
        else:
            out = self.conv(x)
            shortcut = self.shortcut(x)

        out = torch.cat([out,shortcut],1)

        out = channel_shuffle(out,2)
        return out

class ShuffleNet_V2(nn.Module):
    def __init__(self,s=0.5):
        super(ShuffleNet_V2,self).__init__()

        repeat = [3,7,3]
        channel_list = {1:48,2:116,3:176,4:244}
        channels = channel_list[int(s*2)]
        final = 1024 if s<2 else 2048
        def make_layers(in_channels,out_channels,stage):
            layers = []
            layers.append(shufflenet_unit(in_channels,out_channels,2))
            for _ in range(repeat[stage]):
                layers.append(shufflenet_unit(out_channels,out_channels,1))

            return nn.Sequential(*layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,24,3,2,1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )

        self.stage1 = make_layers(24,channels,0)
        self.stage2 = make_layers(channels,channels*2,1)
        self.stage3 = make_layers(channels*2,channels*4,2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels*4,final,1,1),
            nn.BatchNorm2d(final),
def channel_shuffle(input,group):
    b,c,h,w = input.size()
    ranges = c // group
    out = input.view(b,group,ranges,h,w)
    out = out.permute(0,2,1,3,4)
    out = out.reshape(b,c,h,w)
    return out

class shufflenet_unit(nn.Module):
    def __init__(self,in_channels,out_channels,stride,ratio=0.5):
        super(shufflenet_unit,self).__init__()

        if stride == 1:
            in_c = int(in_channels*ratio)
            out_c = int(out_channels*ratio)
        else :
            in_c = in_channels
            out_c = out_channels//2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c,in_c,3,2,1,groups=in_c),
                nn.BatchNorm2d(in_c),

                nn.Conv2d(in_c,out_c,1,1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.s = stride
        self.split = in_c
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,1,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),

            nn.Conv2d(out_c,out_c,3,stride,1,groups=out_c),
            nn.BatchNorm2d(out_c),

            nn.Conv2d(out_c,out_c,1,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )


    def forward(self,x):
        if self.s == 1:
            out = self.conv(x[:,:self.split,:,:])
            shortcut = x[:,self.split:,:,:]
        else:
            out = self.conv(x)
            shortcut = self.shortcut(x)

        out = torch.cat([out,shortcut],1)

        out = channel_shuffle(out,2)
        return out

class ShuffleNet_V2(nn.Module):
    def __init__(self,s=0.5):
        super(ShuffleNet_V2,self).__init__()

        repeat = [3,7,3]
        channel_list = {1:48,2:116,3:176,4:244}
        channels = channel_list[int(s*2)]
        final = 1024 if s<2 else 2048
        def make_layers(in_channels,out_channels,stage):
            layers = []
            layers.append(shufflenet_unit(in_channels,out_channels,2))
            for _ in range(repeat[stage]):
                layers.append(shufflenet_unit(out_channels,out_channels,1))

            return nn.Sequential(*layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,24,3,2,1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )

        self.stage1 = make_layers(24,channels,0)
        self.stage2 = make_layers(channels,channels*2,1)
        self.stage3 = make_layers(channels*2,channels*4,2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels*4,final,1,1),
            nn.BatchNorm2d(final),
            nn.AdaptiveAvgPool2d(1)
        )

        self.FC = nn.Sequential(
            nn.Linear(final,1000)
        )

    def forward(self,x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.conv2(out)

        out = out.view(out.size(0),-1)
        out = self.FC(out)
        return out


summary(ShuffleNet_V2(),(3,224,224))