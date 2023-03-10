import keras

def conv(input,chennel,kernel_sizes,strides=2,padding='valid',is_relu=True,is_bn=True):
  x = input
  x = keras.layers.Conv2D(chennel,kernel_sizes,strides,padding=padding)(x)
  if is_bn:
    x = keras.layers.BatchNormalization()(x)
  if is_relu:
    x = keras.activations.relu(x)
  return x

def se_resnet50(ratio=16):
  inputs = keras.Input(shape=[224,224,3])
  x= inputs
  initial_features=64

  x = conv(x,initial_features,7,2,'same')
  x = keras.layers.MaxPool2D(3,2,'same')(x)

  repeat = [3,4,6,3]
  for i in range(4):
    for j in range(repeat[i]):
      shortcut = x
      strides = 1
      if j==0:  
        if i != 0:
          strides = 2
        shortcut = conv(shortcut,initial_features*2**(i+2),1,strides,'same',False)
        
      x = conv(x,initial_features*2**i,1,strides,'same')
      x = conv(x,initial_features*2**i,3,1,'same')
      x = conv(x,initial_features*2**(i+2),1,1,'same',False)

      se = keras.layers.GlobalAvgPool2D()(x)
      ex = keras.layers.Dense(initial_features*2**(i+2)/ratio)(se)
      ex = keras.activations.relu(ex)
      ex = keras.layers.Dense(initial_features*2**(i+2))(ex)
      ex = keras.activations.sigmoid(ex)

      scale = ex*x

      x = keras.layers.add([scale,shortcut])
      x = keras.activations.relu(x)
  
  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dense(1000)(x)
  x = keras.activations.relu(x)

  return keras.Model(inputs=[inputs], outputs=[x], name=f'SE-ResNet50')

model = se_resnet50()
model.summary()