import keras

def conv(input,features,kernel_size=3,strides = 1,padding='same',is_relu=True,is_bn=False):
  x= keras.layers.Conv2D(features,kernel_size,strides,padding)(input)
  if is_bn:
    x = keras.layers.BatchNormalization()(x)
  if is_relu:
    x = keras.activations.relu(x)
  return x

def unet2(n_levels,DSV=True, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
  inputs = keras.layers.Input(shape=(400, 400, in_channels))
  x = inputs

  #인코더부분
  skips = []

  for _ in range(n_levels):
    skips.append(list())

  for level in range(n_levels):
    if level != 0 :
      x = keras.layers.MaxPool2D(pooling_size)(x)
    for _ in range(n_blocks):
      x = conv(x,initial_features * 2 ** level,3,1,'same')
    skips[level].append(x)


  #스킵 생성 부분
  for i in range(1,n_levels):
    for level in range(n_levels-i):
      list_concat = []

      for row in range(i):
        list_concat.append(skips[level][row])
      
      x = skips[level+1][i-1]
      x = keras.layers.UpSampling2D(pooling_size, interpolation='bilinear')(x) 
      list_concat.append(x)

      x = keras.layers.Concatenate()(list_concat)
      for _ in range(n_blocks):
        x = conv(x,initial_features * 2 ** level,3,1,'same')
      skips[level].append(x)
          
  # 출력부분
  result = []
  if DSV:
    for i in range(1,n_levels):
      result.append(keras.layers.Conv2D(out_channels, kernel_size=1, padding='same')(skips[0][i]))
  else:
    result.append(keras.layers.Conv2D(out_channels, kernel_size=1, padding='same')(skips[0][-1]))
  
  for i in range(len(result)):
    if out_channels == 1:
      result[i] = keras.activations.sigmoid(result[i])
    else:
      result[i] = keras.activations.softmax(result[i])
  
  #모델 이름 설정
  output_name=f'UNET2-L{n_levels}-F{initial_features}'
  if DSV:
    output_name+='-DSV'

  return keras.Model(inputs=[inputs], outputs=result, name=output_name)

model = unet2(5,True)