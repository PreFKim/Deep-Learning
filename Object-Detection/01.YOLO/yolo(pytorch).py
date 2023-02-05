import torch.nn as nn
from torchsummary import summary

class yolo(nn.Module):
    def __init__(self,s=7,b=2,c=20):
        super(yolo,self).__init__()
        self.s = s
        self.b = b
        self.c = c

        def cbl(in_channels,out_channels,kernel_size,stride,padding=0):
            layers = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
            return layers
        self.conv1 = cbl(3,192,7,2,3)
        self.conv2 = cbl(192,256,3,1,1)
        self.conv3 = nn.Sequential(
            cbl(256,128,1,1),
            cbl(128,256,3,1,1),
            cbl(256,256,1,1),
            cbl(256,512,3,1,1)
        )
        self.conv4 = nn.Sequential(
            cbl(512,256,1,1),
            cbl(256,512,3,1,1),
            cbl(512,256,1,1),
            cbl(256,512,3,1,1),
            cbl(512,256,1,1),
            cbl(256,512,3,1,1),
            cbl(512,256,1,1),
            cbl(256,512,3,1,1),

            cbl(512,512,1,1),
            cbl(512,1024,3,1,1)
        )
        self.conv5 = nn.Sequential(
            cbl(1024,512,1,1),
            cbl(512,1024,3,1,1),
            cbl(1024,512,1,1),
            cbl(512,1024,3,1,1),

            cbl(1024,1024,1,1),
            cbl(1024,1024,3,2,1)
        )
        self.conv6 = nn.Sequential(
            cbl(1024,1024,3,1,1),
            cbl(1024,1024,3,1,1)
        )

        self.FC = nn.Sequential(
            nn.Linear(s*s*1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,s*s*(b*5+c))
        )
        self.pool = nn.MaxPool2d(2,2)

    def forward(self,x):
        s = self.s
        b = self.b
        c = self.c
        out = self.conv1(x)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = self.conv4(out)
        out = self.pool(out)
        out = self.conv5(out)
        out = self.conv6(out)

        out = out.view(out.size(0),-1)


        out = self.FC(out)

        out = out.view(out.size(0),s,s,(b*5+c))


print(summary(yolo(),(3,448,448)))