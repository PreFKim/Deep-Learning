import torch.nn as nn
from torchsummary import summary

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5,self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(1,6,5,1),
            nn.ReLU()
        )
        self.s2 = nn.AvgPool2d(2,2)
        self.c3 = nn.Sequential(
            nn.Conv2d(6,16,5,1),
            nn.ReLU()
        )
        self.s4 = nn.AvgPool2d(2,2)

        self.c5 = nn.Sequential(
            nn.Conv2d(16,120,5,1),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120,84),
            nn.ReLU(),
            
        )

        self.fc2 = nn.Sequential(
            nn.Linear(84,10),
            nn.Softmax()
            
        )
    def forward(self,x):
        c1 = self.c1(x)
        s2 = self.s2(c1)
        c3 = self.c3(s2)
        s4 = self.s4(c3)
        c5 = self.c5(s4)
        c5 = c5.view(c5.size(0),-1)
        fc1 = self.fc1(c5)
        fc2 = self.fc2(fc1)

        return fc2
summary(LeNet_5(),(1,32,32))