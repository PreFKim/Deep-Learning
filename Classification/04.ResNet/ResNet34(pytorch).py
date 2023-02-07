import torch.nn as nn
from torchsummary import summary 


class basic_block(nn.Module):
    def __init__(self,i,o,s,stage):
        super(basic_block,self).__init__()

        
        self.conv1 = nn.Conv2d(i,o,3,s,1)
        self.bn = nn.BatchNorm2d(o)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(o,o,3,1,'same')

        if s == 2:
          self.identity = nn.Sequential(
              nn.Conv2d(i,o,1,2),
              nn.BatchNorm2d(o)
          )
        else:
          self.identity = nn.Sequential()

    def forward(self,x):
        identity = self.identity(x)

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet34(nn.Module):
    def __init__(self,e=1,num_layers=[3,4,6,3]):
        super(ResNet34,self).__init__()
        def n_blocks(i,o,s,stage):
            layers = []
            layers.append(basic_block(i,o,s,stage))

            for _ in range(1,num_layers[stage]):
                layers.append(basic_block(o*e,o,1,stage))

            return nn.Sequential(*layers)

        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )

        self.stage1 = n_blocks(64,64,1,0)
        self.stage2 = n_blocks(64*e,128,2,1)
        self.stage3 = n_blocks(128*e,256,2,2)
        self.stage4 = n_blocks(256*e,512,2,3)

        self.F = nn.AdaptiveAvgPool2d(1)

        self.FC = nn.Sequential(
            nn.Linear(512*e,1000),
            nn.Softmax(1)
        )

    def forward(self,x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.F(out)

        out = out.view(out.size(0),-1)

        out = self.FC(out)
        
        return out

summary(ResNet34(),(3,224,224))