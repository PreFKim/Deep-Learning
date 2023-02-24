import keras

def inverted_residual_block(input,t,c,n,s):
    x = input

    for i in range(n):
        if i == 0 :
            stride = s
        else :
            stride = 1
            shortcut = x

        x = keras.layers.Conv2D(x.shape[-1]*t,1,1,'same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.relu6(x)

        x = keras.layers.DepthwiseConv2D(3,stride,'same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.activations.relu6(x)

        x = keras.layers.Conv2D(c,1,1,'same')(x)
        x = keras.layers.BatchNormalization()(x)


        if (stride == 1) and (i != 0):
            x = keras.layers.Add()([shortcut,x])

    return x


def mobilenetv2(w=1.0):
    input = keras.layers.Input((224,224,3))

    x = keras.layers.Conv2D(int(32*w),3,2,'same')(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu6(x)

    x = inverted_residual_block(x,1,int(16*w),1,1)
    x = inverted_residual_block(x,6,int(24*w),2,2)
    x = inverted_residual_block(x,6,int(32*w),3,2)
    x = inverted_residual_block(x,6,int(64*w),4,2)
    x = inverted_residual_block(x,6,int(96*w),3,1)
    x = inverted_residual_block(x,6,int(160*w),3,2)
    x = inverted_residual_block(x,6,int(320*w),1,1)


    x = keras.layers.Conv2D(int(1280*w),1,1,'same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu6(x)

    x = keras.layers.GlobalAvgPool2D()(x)

    x = keras.layers.Dense(1000)(x)
    x = keras.activations.softmax(x)

    return keras.Model(inputs=input,outputs=x,name='MobileNetV2')

model = mobilenetv2()
model.summary()