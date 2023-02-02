def conv(input,chennel,kernel_sizes,strides=2,padding='valid',is_relu=True,is_bn=True):
  x = input
  x = keras.layers.Conv2D(chennel,kernel_sizes,strides,padding=padding)(x)
  if is_bn:
    x = keras.layers.BatchNormalization()(x)
  if is_relu:
    x = keras.activations.relu(x)
  return x

def resnet34():
  inputs = keras.Input(shape=[224,224,3])

  initial_features = 64

  x= inputs
  x = conv(x,initial_features,7,2,'same')
  x = keras.layers.MaxPool2D(3,2,'same')(x)

  repeat = [3,4,6,3]
  for i in range(4):
    for j in range(repeat[i]):
      if j==0 and i != 0: 
        shortcut = conv(x,initial_features*2**i,1,2,'same',False)
        strides = 2
      else:
        shortcut = x
        strides = 1
        
      x = conv(x,initial_features*2**i,3,strides,'same')
      x = conv(x,initial_features*2**i,3,1,'same',False)
      x = keras.layers.add([x,shortcut])
      x = keras.activations.relu(x)
  
  x = keras.layers.GlobalAvgPool2D()(x)
  x = keras.layers.Dense(1000)(x)
  x = keras.activations.softmax(x)

  return keras.Model(inputs=[inputs], outputs=[x], name=f'ResNet34')

model = resnet34()
model.summary()