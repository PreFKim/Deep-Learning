import keras

#CPU의 쓰레드가 8임을 고려해 8채널로 나누어 떨어지게 하면 효율적임
def _make_divisible(v, divisor=8, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return int(new_v)


def h_swish(x):
    return x * (keras.activations.relu6(x+3.0) / 6.0)

def inverted_residual_block(input,e,o,k,s,re=False,se=False):

    in_channels = input.shape[-1]
    middle = int(in_channels*e)

    if re:
        nonlinear = keras.activations.relu6
    else :
        nonlinear = h_swish

    x = keras.layers.Conv2D(middle,1,1,'same')(input)
    x = keras.layers.BatchNormalization()(x)
    x = nonlinear(x)

    x = keras.layers.DepthwiseConv2D(k,s,'same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = nonlinear(x)

    if se:
        se = keras.layers.GlobalAvgPool2D()(x)
        ex = keras.layers.Dense(_make_divisible(middle//4))(se)
        ex = keras.activations.relu(ex)
        ex = keras.layers.Dense(middle)(ex)
        ex = h_swish(ex)
        x = ex*x

    x = keras.layers.Conv2D(o,1,1,'same')(x)
    x = keras.layers.BatchNormalization()(x)


    if in_channels == o and s==1:
        x = keras.layers.Add()([input,x])

    return x


def mobilenetv3(ver=0,w=1):
    input = keras.layers.Input((224,224,3))

    large = [
        [1,16,3,1,False,False],
        [4,24,3,2,False,False],
        [3,24,3,1,False,False],
        [3,40,5,2,False,True],
        [3,40,5,1,False,True],
        [3,40,5,1,False,True],
        [6,80,3,2,True,False],
        [2.5,80,3,1,True,False],
        [2.4,80,3,1,True,False],
        [2.4,80,3,1,True,False],
        [6,112,3,1,True,True],
        [6,112,3,1,True,True],
        [6,160,5,2,True,True],
        [6,160,5,1,True,True],
        [6,160,5,1,True,True]
    ]

    small = [
        [1,16,3,2,False,True],
        [4,24,3,2,False,False],
        [11.0/3.0,24,3,1,False,False],
        [4,40,5,2,True,True],
        [6,40,5,1,True,True],
        [6,40,5,1,True,True],
        [3,48,5,1,True,True],
        [3,48,5,1,True,True],
        [6,96,5,2,True,True],
        [6,96,5,1,True,True],
        [6,96,5,1,True,True],
    ]

    if ver == 0:
        stack = large
        last = 1280
        name = 'Large'
    else :
        stack = small
        last = 1024
        name = 'Small'

    x = keras.layers.Conv2D(_make_divisible(16*w),3,2,'same')(input)
    x = keras.layers.BatchNormalization()(x)
    x = h_swish(x)

    for i in range(len(stack)):
        x = inverted_residual_block(x,stack[i][0],_make_divisible(stack[i][1]*w),stack[i][2],stack[i][3],stack[i][4],stack[i][5])

    x = keras.layers.Conv2D(int(x.shape[-1]*6),1,1,'same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = h_swish(x)
    
    x = keras.layers.GlobalAvgPool2D()(x)


    x = keras.layers.Dense(last)(x)
    x = h_swish(x)

    x = keras.layers.Dense(1000)(x)
    x = keras.activations.softmax(x)

    return keras.Model(inputs=input,outputs=x,name=f'MobileNetV3-{name}')

model = mobilenetv3()
model.summary()
