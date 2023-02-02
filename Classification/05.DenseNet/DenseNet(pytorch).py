import torch
import torch.nn as nn
from torchsummary import summary 

class bottleneck_layer(nn.Module):
  def __init__(self,i,g):
    super(bottleneck_layer,self).__init__()

    self.bn1 = nn.BatchNorm2d(i)
    self.conv1 = nn.Conv2d(i,4*g,1,1,'same')
    self.bn2 = nn.BatchNorm2d(4*g)
    self.conv2 = nn.Conv2d(4*g,g,3,1,'same')
    self.relu = nn.ReLU()

  def forward(self,x):
    out = self.bn1(x)
    out = self.relu(out)
    out = self.conv1(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)

    return torch.cat([x,out],1)

class transition_layer(nn.Module):
  def __init__(self,i,compression):
    super(transition_layer,self).__init__()
    self.bn = nn.BatchNorm2d(i)
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(i,int(i*compression),1,1,'same')
    self.pool = nn.AvgPool2d(2,2)
  
  def forward(self,x):
    out = self.bn(x)
    out = self.relu(out)
    out = self.conv(out)
    out = self.pool(out)

    return out



class DenseNet(nn.Module):
  def __init__(self,growth_rate=32,compression=0.5,num_layers=[6,12,32,32]):
    super(DenseNet,self).__init__()

    def dense_block(in_channels,n):
      layer = []
      for i in range(n):
        layer.append(bottleneck_layer(in_channels+i*growth_rate,growth_rate))
      return nn.Sequential(*layer)
    
    out_first = growth_rate*2
    out_stage1 = out_first+growth_rate*num_layers[0]
    out_stage2 = int(out_stage1*compression)+growth_rate*num_layers[1]
    out_stage3 = int(out_stage2*compression)+growth_rate*num_layers[2]
    out_stage4 = int(out_stage3*compression)+growth_rate*num_layers[3]

    self.first = nn.Sequential(
        nn.Conv2d(3,out_first,7,2,3),
        nn.BatchNorm2d(out_first),
        nn.ReLU(),
        nn.MaxPool2d(3,2,1)
    )

    self.stage1 = dense_block(out_first,num_layers[0])
    self.transition1 = transition_layer(out_stage1,compression)

    self.stage2 = dense_block(int(out_stage1*compression),num_layers[1])
    self.transition2 = transition_layer(out_stage2,compression)

    self.stage3 = dense_block(int(out_stage2*compression),num_layers[2])
    self.transition3 = transition_layer(out_stage3,compression)

    self.stage4 = dense_block(int(out_stage3*compression),num_layers[3])
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.FC = nn.Sequential(
        nn.Linear(out_stage4,1000),
        nn.Softmax()
    )

  def forward(self,x):
    out = self.first(x)

    out = self.stage1(out)
    out = self.transition1(out)

    out = self.stage2(out)
    out = self.transition2(out)

    out = self.stage3(out)
    out = self.transition3(out)

    out = self.stage4(out)
    out = self.pool(out)
    out = out.view(out.size(0), -1)
    out = self.FC(out)
    return out

summary(DenseNet(),(3,224,224))
