def conv(x,f,k,s,n):
  for i in range(n):
    x = keras.layers.Conv2D(f,k,s,padding='same')(x)
    x = keras.activations.relu(x)
  return x

def vgg16():
  x = keras.layers.Input((224,224,3))

  L1 = conv(x,64,3,1,2)
  
  L2 = keras.layers.MaxPool2D(2,2)(L1)
  L2 = conv(L2,128,3,1,2)

  L3 = keras.layers.MaxPool2D(2,2)(L2)
  L3 = conv(L3,256,3,1,3)

  L4 = keras.layers.MaxPool2D(2,2)(L3)
  L4 = conv(L4,512,3,1,3)

  L5 = keras.layers.MaxPool2D(2,2)(L4)
  L5 = conv(L5,512,3,1,3)

  F = keras.layers.MaxPool2D(2,2)(L5)
  F = keras.layers.Flatten()(F)

  FC1 = keras.layers.Dense(4096)(F)
  FC1 = keras.activations.relu(FC1)
  
  FC2 = keras.layers.Dense(4096)(FC1)
  FC2 = keras.activations.relu(FC2)

  FC3 = keras.layers.Dense(1000)(FC2)
  FC3 = keras.activations.relu(FC3)
  
  output = keras.activations.softmax(FC3)

  return keras.Model(inputs=x,outputs=output,name='VGG16')
model = vgg16()
model.summary()