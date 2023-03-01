import keras

def swish(x):
    return x* keras.activations.sigmoid(x)

def MBConv(input,e,kernel_size,out_channels,stride):
  x = input

  in_channels = x.shape[-1]

  x = keras.layers.Conv2D(in_channels*e,1,1,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = swish(x)

  x = keras.layers.DepthwiseConv2D(kernel_size,stride,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = swish(x)

  se = keras.layers.GlobalAvgPool2D()(x)
  ex = keras.layers.Dense(int(in_channels*e/4))(se)
  ex = swish(ex)
  ex = keras.layers.Dense(in_channels*e)(ex)
  ex = keras.activations.sigmoid(ex)
  scale = ex*x

  x = swish(scale)

  x = keras.layers.Conv2D(out_channels,1,1,'same')(x)
  x = keras.layers.BatchNormalization()(x)

  if stride == 1 and in_channels == out_channels:
    #x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.add([input,x])
  return x

def EfficientNet_B0():
  input = keras.layers.Input((224,224,3)) 
  x = input

  x = keras.layers.Conv2D(32,3,2,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = swish(x)

  x = MBConv(x,1,3,16,1)

  x = MBConv(x,6,3,24,1)
  x = MBConv(x,6,3,24,2)

  x = MBConv(x,6,5,40,1)
  x = MBConv(x,6,5,40,2)

  x = MBConv(x,6,3,80,1)
  x = MBConv(x,6,3,80,1)
  x = MBConv(x,6,3,80,2)

  x = MBConv(x,6,5,112,1)
  x = MBConv(x,6,5,112,1)
  x = MBConv(x,6,5,112,1)

  x = MBConv(x,6,5,192,1)
  x = MBConv(x,6,5,192,1)
  x = MBConv(x,6,5,192,1)
  x = MBConv(x,6,5,192,2)

  x = MBConv(x,6,3,320,1)

  x = keras.layers.Conv2D(1280,1,1,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = swish(x)

  x = keras.layers.GlobalAvgPool2D()(x)
  x = keras.layers.Dense(1280)(x)
  x = keras.activations.softmax(x)
  
  return keras.Model(inputs=input , outputs=x,name = 'EfficientNet-B0')

model = EfficientNet_B0()
model.summary()
  