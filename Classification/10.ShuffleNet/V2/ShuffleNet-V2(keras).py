import keras
import keras.backend as K
import keras.backend as K


def shuffle(input,group):
  _,h,w,c = input.shape
  ranges = c // group

  x = K.reshape(input, [-1, h, w, group, ranges])
  x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  
  x = K.reshape(x, [-1, h, w, c])
  return x


def shufflenet_unit(input,channels,stride,ratio=0.5):
  if stride == 1:
    out_c = int(channels*ratio)
    split = int(input.shape[-1] * ratio)

    x = input[:,:,:,:split]
    shortcut = input[:,:,:,split:]
  else :
    out_c = channels//2

    x = input
    shortcut = keras.layers.DepthwiseConv2D(3,2,'same')(input)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    shortcut = keras.layers.Conv2D(out_c,1,1,'same')(shortcut)
    shortcut = keras.layers.BatchNormalization()(shortcut)
    shortcut = keras.activations.relu(shortcut)

  x = keras.layers.Conv2D(out_c,1,1,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)

  x = keras.layers.DepthwiseConv2D(3,stride,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  
  x = keras.layers.Conv2D(out_c,1,1,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)

  x = keras.layers.Concatenate()([x,shortcut])
  x = shuffle(x,2)
  return x
  
  

def shufflenet_v2(s=0.5):
  input = keras.layers.Input((224,224,3))

  repeat = [3,7,3]
  channel_list = {1:48,2:116,3:176,4:244}

  channels = channel_list[int(s*2)]

  x = keras.layers.Conv2D(24,3,2,'same')(input)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)

  x = keras.layers.MaxPool2D(3,2,'same')(x)

  for i in range(len(repeat)):
    x = shufflenet_unit(x,channels,2)
    for _ in range(repeat[i]):
      x = shufflenet_unit(x,channels,1)
    channels *= 2
  
  x = keras.layers.Conv2D(1024 if s<2 else 2048,1,1,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)

  x = keras.layers.GlobalAvgPool2D()(x)

  x = keras.layers.Dense(1000)(x)
  x = keras.activations.softmax(x)

  return keras.Model(inputs=input,outputs=x,name='ShuffleNet_V2')

model = shufflenet_v2()
model.summary()