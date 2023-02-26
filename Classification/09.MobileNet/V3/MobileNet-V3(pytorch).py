import torch.nn as nn
from torchsummary import summary

#CPU의 쓰레드가 8임을 고려해 8채널로 나누어 떨어지게 하면 효율적임
def _make_divisible(v, divisor=8, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return int(new_v)

class h_swish(nn.Module):
    def __init__(self):
        super(h_swish,self).__init__()
        self.relu6 = nn.ReLU6()
    
    def forward(self,x):
        return x*(self.relu6(x+3)/6)


class inverted_residual_block(nn.Module):
    def __init__(self,i,t,o,k,s,re=False,se=False):
        super(inverted_residual_block,self).__init__()

        expansion = int(i*t)
        if re:
            nonlinear = nn.ReLU6()
        else:
            nonlinear = h_swish()

        self.se = se

        self.conv = nn.Sequential(
            nn.Conv2d(i,expansion,1,1),
            nn.BatchNorm2d(expansion),
            nonlinear
        )

        self.dconv = nn.Sequential(
            nn.Conv2d(expansion,expansion,k,s,k//2,groups=expansion),
            nn.BatchNorm2d(expansion),
            nonlinear
        )

        self.semodule = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expansion,_make_divisible(expansion//4),1,1),
            nn.ReLU(),
            nn.Conv2d(_make_divisible(expansion//4),expansion,1,1),
            h_swish()
        )

        self.linearconv = nn.Sequential(
            nn.Conv2d(expansion,o,1,1),
            nn.BatchNorm2d(o)
        )

        self.shortcut = (i==o and s==1)

    def forward(self,x):
        out = self.conv(x)
        out = self.dconv(out)

        if self.se :
            out *= self.semodule(out)

        out = self.linearconv(out)

        if self.shortcut :
            out += x

        return out



class mobilenetv3(nn.Module):
    def __init__(self,ver=0,w=1.0):
        super(mobilenetv3,self).__init__()

        large = [
            [1,16,3,1,False,False],
            [4,24,3,2,False,False],
            [3,24,3,1,False,False],
            [3,40,5,2,False,True],
            [3,40,5,1,False,True],
            [3,40,5,1,False,True],
            [6,80,3,2,True,False],
            [2.5,80,3,1,True,False],
            [2.4,80,3,1,True,False],
            [2.4,80,3,1,True,False],
            [6,112,3,1,True,True],
            [6,112,3,1,True,True],
            [6,160,5,2,True,True],
            [6,160,5,1,True,True],
            [6,160,5,1,True,True]
        ]

        small = [
            [1,16,3,2,False,True],
            [4,24,3,2,False,False],
            [11.0/3.0,24,3,1,False,False],
            [4,40,5,2,True,True],
            [6,40,5,1,True,True],
            [6,40,5,1,True,True],
            [3,48,5,1,True,True],
            [3,48,5,1,True,True],
            [6,96,5,2,True,True],
            [6,96,5,1,True,True],
            [6,96,5,1,True,True],
        ]

        in_channels = _make_divisible(16*w)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,in_channels,3,2,1),
            nn.BatchNorm2d(int(16*w)),
            nn.ReLU6()
        )

        if ver == 0:
            stack = large
            last = 1280
        else :
            stack = small
            last = 1024
        

        layers = []


        for i in range(len(stack)):    
            out_channels = _make_divisible(stack[i][1]*w)
            layers.append(inverted_residual_block(in_channels,stack[i][0],out_channels,stack[i][2],stack[i][3],stack[i][4],stack[i][5]))
            in_channels = out_channels

        self.stack = nn.Sequential(*layers)

        self.last = nn.Sequential(
            nn.Conv2d(in_channels,out_channels*6,1,1),
            nn.BatchNorm2d(out_channels*6),
            h_swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels*6,last,1,1),
            h_swish(),
            nn.Conv2d(last,1000,1,1)
        )


    def forward(self,x):

        out = self.conv1(x)
        out = self.stack(out)
        out = self.last(out)
        out = out.view(out.size(0),-1)

        return out
summary(mobilenetv3(),(3,224,224))