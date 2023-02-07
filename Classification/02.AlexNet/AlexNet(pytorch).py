import torch.nn as nn
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(3,96,11,4),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,1,'same'),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1,'same'),
            nn.ReLU()
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(384,384,3,1,'same'),
            nn.ReLU()
        )

        self.l5 = nn.Sequential(
            nn.Conv2d(384,256,3,1,'same'),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(256,4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5)
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

        pool = self.pool(l5)
        pool = pool.view(pool.size(0),-1)

        fc1 = self.fc1(pool)
        fc2 = self.fc2(fc1)
        output = self.output(fc2)

        return output

summary(AlexNet(),(3,227,227))