import keras
def alexnet():
  x = keras.layers.Input((227,227,3))

  l1 = keras.layers.Conv2D(96,11,4)(x)
  l1 = keras.activations.relu(l1)

  l2 = keras.layers.BatchNormalization()(l1)
  l2 = keras.layers.MaxPool2D(3,2)(l2)
  l2 = keras.layers.Conv2D(256,5,padding='same')(l2)
  l2 = keras.activations.relu(l2)
  
  l3 = keras.layers.BatchNormalization()(l2)
  l3 = keras.layers.MaxPool2D(3,2)(l3)
  l3 = keras.layers.Conv2D(384,3,padding='same')(l3)
  l3 = keras.activations.relu(l3)

  l4 = keras.layers.Conv2D(384,3,padding='same')(l3)
  l4 = keras.activations.relu(l4)

  l5 = keras.layers.Conv2D(256,3,padding='same')(l4)
  l5 = keras.activations.relu(l5)

  f = keras.layers.GlobalMaxPool2D()(l5)

  fc1 = keras.layers.Dense(4096)(f)
  fc1 = keras.activations.relu(fc1)
  fc1 = keras.layers.Dropout(0.5)(fc1)

  fc2 = keras.layers.Dense(4096)(fc1)
  fc2 = keras.activations.relu(fc2)
  fc2 = keras.layers.Dropout(0.5)(fc2)

  output = keras.layers.Dense(1000)(fc2)
  output = keras.activations.softmax(output)

  return keras.Model(inputs=[x], outputs=[output],name ='AlexNet')

model = alexnet()

model.summary()
