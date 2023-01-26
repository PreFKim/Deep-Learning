import keras
import keras.backend as K

def gconv(input,group,channels):
  start_idx = 0
  ranges = input.shape[-1]//group

  group_list=[]
  
  for i in range(group):
    group_list.append(keras.layers.Conv2D(channels//group,1,1,'same')(input[:,:,:,start_idx:start_idx+ranges]))

  if group == 1:
    x = group_list[0]
  else :
    x = keras.layers.Concatenate()(group_list)

  x = keras.layers.BatchNormalization()(x)
  return x

def shuffle(input,group):
  _,h,w,c = input.shape
  ranges = c // group

  x = K.reshape(input, [-1, h, w, group, ranges])
  x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  
  x = K.reshape(x, [-1, h, w, c])
  return x


def shufflenet_unit(input,group,out_channels,stride,i=1):
  if stride == 1:
    shortcut = input
    final = out_channels
    op = keras.layers.Add()
  else :
    shortcut = keras.layers.AvgPool2D(3,2,'same')(input)
    final = out_channels - input.shape[-1]
    op = keras.layers.Concatenate()

  
  x = input
  if i==0:
    x = gconv(x,1,out_channels/4)
  else :
    x = gconv(x,group,out_channels/4)

  x = keras.activations.relu(x)

  x = shuffle(x,group)

  x = keras.layers.DepthwiseConv2D(3,stride,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = gconv(x,group,final)

  x = op([x,shortcut])
  x = keras.activations.relu(x)
  return x
  
  

def shufflenet(g=1,s=1):
  input = keras.layers.Input((224,224,3))

  repeat = [3,7,3]
  channel_list = {1:144,2:200,3:240,4:272,8:384}

  channels = channel_list[g]*s

  x = keras.layers.Conv2D(24,3,2,'same')(input)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)

  x = keras.layers.MaxPool2D(3,2,'same')(x)

  for i in range(len(repeat)):
    x = shufflenet_unit(x,g,channels,2,i)
    for _ in range(repeat[i]):
      x = shufflenet_unit(x,g,channels,1)
    channels *= 2

  x = keras.layers.GlobalAvgPool2D()(x)

  x = keras.layers.Dense(1000)(x)
  x = keras.activations.softmax(x)

  return keras.Model(inputs=input,outputs=x,name='ShuffleNet')

model = shufflenet()
model.summary()