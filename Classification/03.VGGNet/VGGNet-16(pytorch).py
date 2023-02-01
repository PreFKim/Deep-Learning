import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet,self).__init__()
        def conv(i,o,k,s):

            x = nn.Sequential(
                nn.Conv2d(i,o,k,s,1),
                nn.ReLU(),
                nn.Conv2d(o,o,k,s,1),
                nn.ReLU()
            )

            return x
            
                
        self.l1 = conv(3,64,3,1)
        self.l2 = nn.Sequential(
            nn.MaxPool2d(2,2),
            conv(64,128,3,1)
        )
        self.l3 = nn.Sequential(
            nn.MaxPool2d(2,2),
            conv(128,256,3,1),
            nn.Conv2d(256,256,3,1,1)
        )

        self.l4 = nn.Sequential(
            nn.MaxPool2d(2,2),
            conv(256,512,3,1),
            nn.Conv2d(512,512,3,1,1)
        )

        self.l5 = nn.Sequential(
            nn.MaxPool2d(2,2),
            conv(512,512,3,1),
            nn.Conv2d(512,512,3,1,1)
        )

        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(512,4096),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.Linear(4096,1000),
            nn.Softmax()
        )

    def forward(self,x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)
        l4 = self.l4(l3)
        l5 = self.l5(l4)

        fc1 = self.fc1(l5)
        fc2 = self.fc2(fc1)
        output = self.output(fc2)

        return output
        
print(VGGNet())