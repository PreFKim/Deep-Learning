import keras

def RESNET34():
  x = keras.layers.Input((224,224,3))

  l1 = keras.layers.Conv2D(64,7,2,'same')(x)
  l1 = keras.layers.BatchNormalization()(l1)
  l1 = keras.activations.relu(l1)
  

  l2 = keras.layers.MaxPool2D(3,2,'same')(l1)

  for i in range(3):
    shortcut = l2

    l2 = keras.layers.Conv2D(64,3,1,'same')(l2)
    l2 = keras.layers.BatchNormalization()(l2)
    l2 = keras.activations.relu(l2)
    l2 = keras.layers.Conv2D(64,3,1,'same')(l2)
    l2 = keras.layers.BatchNormalization()(l2)

    l2 = keras.layers.Add()([l2,shortcut])
    l2 = keras.activations.relu(l2)
  
  l3 = l2
  for i in range(4):
    if i==0:
      shortcut = keras.layers.Conv2D(128,1,2,'same')(l3)
      shortcut = keras.layers.BatchNormalization()(shortcut)
      stride = 2
    else:
      shortcut = l3
      stride = 1
    
    l3 = keras.layers.Conv2D(128,3,stride,'same')(l3)
    l3 = keras.layers.BatchNormalization()(l3)
    l3 = keras.activations.relu(l3)
    l3 = keras.layers.Conv2D(128,3,1,'same')(l3)
    l3 = keras.layers.BatchNormalization()(l3)

    l3 = keras.layers.Add()([l3,shortcut])
    l3 = keras.activations.relu(l3)
  l4 = l3

  for i in range(6):
    if i==0:
      shortcut = keras.layers.Conv2D(256,1,2,'same')(l4)
      shortcut = keras.layers.BatchNormalization()(shortcut)
      stride = 2
    else:
      shortcut = l4
      stride = 1

    l4 = keras.layers.Conv2D(256,3,stride,'same')(l4)
    l4 = keras.layers.BatchNormalization()(l4)
    l4 = keras.activations.relu(l4)
    l4 = keras.layers.Conv2D(256,3,1,'same')(l4)
    l4 = keras.layers.BatchNormalization()(l4)

    l4 = keras.layers.Add()([l4,shortcut])
    l4 = keras.activations.relu(l4)

  l5 = l4
  for i in range(3):
    if i == 0:
      shortcut = keras.layers.Conv2D(512,1,2,'same')(l5)
      shortcut = keras.layers.BatchNormalization()(shortcut)
      stride = 2
    else:
      shortcut = l5
      stride = 1

    l5 = keras.layers.Conv2D(512,3,stride,'same')(l5)
    l5 = keras.layers.BatchNormalization()(l5)
    l5 = keras.activations.relu(l5)
    l5 = keras.layers.Conv2D(512,3,1,'same')(l5)
    l5 = keras.layers.BatchNormalization()(l5)
    
    l5 = keras.layers.Add()([l5,shortcut])
    l5 = keras.activations.relu(l5)

  output = keras.layers.GlobalAvgPool2D()(l5)
  output = keras.layers.Dense(1000)(output)
  output = keras.activations.softmax(output)

  return keras.Model(inputs=x,outputs=output,name='RESNET-34')

model = RESNET34()

model.summary()
