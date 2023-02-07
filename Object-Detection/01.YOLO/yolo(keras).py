import keras


def conv(input,chennel,kernel_sizes,strides=2,padding='valid',is_lrelu=True,is_bn=True):
  x = input
  x = keras.layers.Conv2D(chennel,kernel_sizes,strides,padding=padding)(x)
  if is_bn:
    x = keras.layers.BatchNormalization()(x)
  if is_lrelu:
    x = keras.activations.leaky_relu(x)
  return x


def YOLO(s=7,b=2,c=20):
    input = keras.layers.Input((448,448,3))

    x = conv(input,192,7,2,'same')
    x = keras.layers.MaxPool2D(2,2)(x)

    x = conv(x,256,3,1,'same')
    x = keras.layers.MaxPool2D(2,2)(x)

    x = conv(x,128,1,1,'same')
    x = conv(x,256,3,1,'same')
    x = conv(x,256,1,1,'same')
    x = conv(x,512,3,1,'same')
    x = keras.layers.MaxPool2D(2,2)(x)

    for _ in range(4):
        x = conv(x,256,1,1,'same')
        x = conv(x,512,3,1,'same')
    x = conv(x,512,1,1,'same')
    x = conv(x,1024,3,1,'same')
    x = keras.layers.MaxPool2D(2,2)(x)

    for _ in range(2):
        x = conv(x,512,1,1,'same')
        x = conv(x,1024,3,1,'same')
    x = conv(x,1024,1,1,'same')
    x = conv(x,1024,3,2,'same')


    x = conv(x,1024,3,1,'same')
    x = conv(x,1024,3,1,'same')

    #FC
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(4096)(x)
    x = keras.activations.leaky_relu(x)

    x = keras.layers.Dense(s*s*(b*5+c))(x)

    x = keras.layers.Reshape((s,s,(b*5+c)))(x)
    print(x.shape)
    localization = x[:,:,:,:b*5]
    classification = keras.activations.softmax(x[:,:,:,b*5:])

    output = keras.layers.Concatenate()([localization,classification])

    return keras.Model(inputs=input,outputs=output,name='YOLO')

model = YOLO()
model.summary()