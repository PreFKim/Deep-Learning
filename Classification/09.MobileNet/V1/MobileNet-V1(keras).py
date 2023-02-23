import keras

def depthwise_sparable_conv(input,oc,dw=0):
  x= input
  x = keras.layers.DepthwiseConv2D(3,dw+1,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)

  x = keras.layers.Conv2D(oc,1,1,'same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  return x

def mobilenet(a=1,p=1):
  input = keras.layers.Input((int(224*p),int(224*p),3))
  
  out_features = 32*a

  x = keras.layers.Conv2D(out_features,3,2,'same')(input)
  x = keras.layers.BatchNormalization()(x)
  x = keras.activations.relu(x)

  for i in range(6):
    if i%2 == 1 or i==0:
      out_features *= 2
    x = depthwise_sparable_conv(x,out_features,i%2)

  for i in range(5):
    x = depthwise_sparable_conv(x,out_features,0)

  for i in range(2):
    x = depthwise_sparable_conv(x , out_features*2,1)
  

  x = keras.layers.GlobalAvgPool2D()(x)
  x = keras.layers.Dense(1000)(x)
  x = keras.activations.softmax(x)

  return keras.Model(inputs=input,outputs=x,name='MobileNet')

model = mobilenet()
model.summary()