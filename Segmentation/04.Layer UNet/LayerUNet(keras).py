import keras

def conv(input,features,kernel_size=3,strides = 1,padding='same',is_relu=True,is_bn=False):
  x= keras.layers.Conv2D(features,kernel_size,strides,padding)(input)
  if is_bn:
    x = keras.layers.BatchNormalization()(x)
  if is_relu:
    x = keras.activations.relu(x)
  return x

def layerUNET(n_levels,DSV=True ,initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
    inputs = keras.layers.Input(shape=(400, 400, in_channels))
    x = inputs

    skips = []
    for _ in range(n_levels):
      skips.append(list())

    #인코더부분
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

        #같은 레벨
        tmp = []
        for j in range(i):
          tmp.append(conv(skips[level][j],initial_features,3,1,'same'))
        if (len(tmp)>1):
          x = keras.layers.concatenate(tmp)
        else :
          x = tmp[0]
        x = conv(x,initial_features,3,1,'same')
        list_concat.append(x)

        #위  레벨
        for row in range(level):
          x = keras.layers.MaxPool2D(pooling_size**(level-row),pooling_size**(level-row))(skips[row][0])
          x = conv(x,initial_features,3,1,'same')
          list_concat.append(x)
        
        #아래 레벨
        for j in range(i):
          x = keras.layers.UpSampling2D(pooling_size**(j+1), interpolation='bilinear')(skips[level+j+1][i-j-1])
          x = conv(x,initial_features,3,1,'same')
          list_concat.append(x)
        

        #Concatenate 부분
        x = keras.layers.concatenate(list_concat)
        for _ in range(1): 
          x = conv(x,initial_features * len(list_concat),3,1,'same',is_bn= True)
        skips[level].append(x)
    
    # 출력부분
    result = []
    
    if DSV:
      for i in range(1,n_levels):
        x = conv(skips[0][i],out_channels,1,1,'same',False,False)
        result.append(x)

      for i in range(1,n_levels):
        x = conv(skips[i][-1],out_channels,3,1,'same',False,False)
        x = keras.layers.UpSampling2D(pooling_size**(i), interpolation='bilinear')(x)
        result.append(x)
    else:
      result.append(x = conv(skips[0][-1],out_channels,1,1,'same',False,False))
    
    for i in range(len(result)):
      if out_channels == 1:
        result[i] = keras.activations.sigmoid(result[i])
      else:
        result[i] = keras.activations.softmax(result[i])

    output_name=f'LayerUNET-L{n_levels}-F{initial_features}'

    if DSV:
      output_name+=f'-DSV'

    return keras.Model(inputs=[inputs], outputs=result, name=output_name)

model = layerUNET(5,True)