import keras

def fire_module(x,s1x1,e1x1,e3x3):
  x = keras.layers.Conv2D(s1x1,1,1,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)

  e1 = keras.layers.Conv2D(e1x1,1,1,'same')(x)
  e1 = keras.layers.BatchNormalization()(e1)
  e1 = keras.activations.relu(e1)

  e2 = keras.layers.Conv2D(e3x3,3,1,'same')(x)
  e2 = keras.layers.BatchNormalization()(e2)
  e2 = keras.activations.relu(e2)

  result = keras.layers.concatenate([e1,e2])

  return result

def Squeezenet():
  input= keras.layers.Input((224,224,3))

  conv1 = keras.layers.Conv2D(96,7,2,'same')(input)
  conv1 = keras.layers.BatchNormalization()(conv1)
  conv1 = keras.activations.relu(conv1)

  maxpool = keras.layers.MaxPool2D(3,2,'same')(conv1)

  fire2 = fire_module(maxpool,16,64,64)

  fire3 = fire_module(fire2,16,64,64)

  fire4 = fire_module(keras.layers.Add()([fire2+fire3]),32,128,128)

  maxpool = keras.layers.MaxPool2D(3,2,'same')(fire4)

  fire5 = fire_module(maxpool,32,128,128)
  fire6 = fire_module(keras.layers.Add()([maxpool,fire5]),48,192,192)
  fire7 = fire_module(fire6,48,192,192)
  fire8 = fire_module(keras.layers.Add()([fire6,fire7]),64,256,256)

  maxpool = keras.layers.MaxPool2D(3,2,'same')(fire8)

  fire9 = fire_module(maxpool,64,256,256)

  dropout = keras.layers.Dropout(0.5)(fire9)

  output = keras.layers.Conv2D(1000,1,1,'same')(keras.layers.Add()([maxpool,dropout]))
  output = keras.layers.GlobalAvgPool2D()(output)
  output = keras.activations.softmax(output)

  return keras.Model(inputs=input,outputs=output, name='SqueezeNet')

model = Squeezenet()
model.summary()