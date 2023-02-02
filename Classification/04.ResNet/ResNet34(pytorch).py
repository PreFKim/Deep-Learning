import torch.nn as nn
from torchsummary import summary 

class basic_block(nn.Module):
    def __init__(self,i,o,s):
        super(basic_block,self).__init__()
        self.s = s
        
        self.conv1 = nn.Conv2d(i,o,3,s,1)
        self.bn = nn.BatchNorm2d(o)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(o,o,3,1,'same')

        self.identity = nn.Conv2d(i,o,1,2)

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.s == 2:
            identity = self.identity(identity)
            identity = self.bn(identity)
        

        out += identity
        out = self.relu(out)

        return out

#64 128

#128 128



class ResNet34(nn.Module):
    def __init__(self,num_layers=[3,4,6,3]):
        super(ResNet34,self).__init__()
        def n_blocks(i,o,s,n):
            layers = []
            layers.append(basic_block(i,o,s))

            for _ in range(1,n):
                layers.append(basic_block(o,o,1))

            return nn.Sequential(*layers)

        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )

        self.stage1 = n_blocks(64,64,1,num_layers[0])
        self.stage2 = n_blocks(64,128,2,num_layers[1])
        self.stage3 = n_blocks(128,256,2,num_layers[2])
        self.stage4 = n_blocks(256,512,2,num_layers[3])

        self.F = nn.AdaptiveAvgPool2d(1)

        self.FC = nn.Sequential(
            nn.Linear(512,1000),
            nn.Softmax()
        )



        
            

    def forward(self,x):
        print('1')
        out = self.conv1(x)
        print(out.shape)
        out = self.stage1(out)
        print('1')
        out = self.stage2(out)
        print('1')
        out = self.stage3(out)
        print('1')
        out = self.stage4(out)
        print('1')

        out = self.F(out)

        out = out.view(out.size(0),-1)

        out = self.FC(out)
        

        return out

summary(ResNet34(),(3,224,224))