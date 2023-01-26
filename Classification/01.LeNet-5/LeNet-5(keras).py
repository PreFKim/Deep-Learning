import keras

def lenet_5():
  x = keras.layers.Input((32,32,1))

  C1 = keras.layers.Conv2D(6,5)(x)
  C1 = keras.activations.relu(C1)

  S2 = keras.layers.AveragePooling2D(2,2)(C1)

  C3 = keras.layers.Conv2D(16,5)(S2)
  C3 = keras.activations.relu(C3)

  S4 = keras.layers.AveragePooling2D(2,2)(C3)
  
  #C5 = keras.layers.Conv2D(120,5)(S4)
  
  F = keras.layers.Flatten()(S4)
  C5 = keras.layers.Dense(120)(F)

  C5 = keras.activations.relu(C5)

  #F = keras.layers.Flatten()(C5)
  
  FC1 = keras.layers.Dense(84)(F)
  FC1 = keras.activations.relu(FC1)

  output = keras.layers.Dense(10)(FC1)
  output = keras.activations.softmax(output)

  return keras.Model(inputs=x,outputs=output,name='LeNet_5')

model = lenet_5()


model.summary()