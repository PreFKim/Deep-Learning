import keras

def DenseNet(growth_rate=32,compression=0.5):
  num_conv=[6,12,32,32]
  input = keras.layers.Input((224,224,3))

  
  x = keras.layers.Conv2D(growth_rate*2,7,2,'same')(input)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = keras.layers.MaxPool2D(3,2,'same')(x)

  for i in range(4):
    block = [x]
    for j in range(num_conv[i]):

      x = keras.layers.BatchNormalization()(x)
      x = keras.activations.relu(x)
      x = keras.layers.Conv2D(4*growth_rate,1,1,'same')(x)

      x = keras.layers.BatchNormalization()(x)
      x = keras.activations.relu(x)
      x = keras.layers.Conv2D(growth_rate,3,1,'same')(x)

      block.append(x)
      x = keras.layers.concatenate(block)

    if (i < 3):
      x = keras.layers.BatchNormalization()(x)
      x = keras.activations.relu(x)
      x = keras.layers.Conv2D(int(compression*x.shape[-1]),1,1,'same')(x)
      x = keras.layers.AvgPool2D(2,2)(x)
    else :

      x = keras.layers.GlobalAvgPool2D()(x)
      x = keras.layers.Dense(1000)(x)
      x = keras.activations.softmax(x)
    
  return keras.Model(inputs=input,outputs=x,name='DenseNet')
model = DenseNet()
model.summary()
