import torch.nn as nn
from torchsummary import summary


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet,self).__init__()
        def conv(i,o,k,s,n):
            layers = []

            for count in range(n):
                if count == 0:
                    layers.append(nn.Conv2d(i,o,k,s,'same'))
                else:
                    layers.append(nn.Conv2d(o,o,k,s,'same'))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2,2))
            return nn.Sequential(*layers)
            
                
        self.l1 = conv(3,64,3,1,2)
        self.l2 = conv(64,128,3,1,2)
        self.l3 = conv(128,256,3,1,3)

        self.l4 = conv(256,512,3,1,3)

        self.l5 = conv(512,512,3,1,3)

        self.fc1 = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(4096,1000),
            nn.Softmax(1)
        )

    def forward(self,x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        l5 = self.l5(l4)
        l5 = l5.view(l5.size(0),-1)

        fc1 = self.fc1(l5)
        fc2 = self.fc2(fc1)
        output = self.output(fc2)

        return output
        
summary(VGGNet(),(3,224,224))