import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(3,96,11,4,0),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,1,1),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3,2),
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU()
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU()
        )

        self.l5 = nn.Sequential(
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
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

model = AlexNet()
