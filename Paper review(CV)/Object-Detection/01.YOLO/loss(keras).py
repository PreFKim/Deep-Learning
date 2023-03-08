import keras.backend as K

def loss_function(y_true,y_pred,lambda_coord=5,lambda_noobj=0.5):
    #이때 y_true 는 (-1,s,s,5+c)로 들어오고 y pred는 (-1,s,s,b*5+c)로 들어온다.
    #(x,y,w,h,confidence)*b,classes(c)

    y_true_f = K.reshape(y_true,(-1,s*s,5+c))
    y_pred_f = K.reshape(y_pred,(-1,s*s,b*5+c))

    batch_size = y_true.shape[0]
    s = y_true.shape[1]
    b = int((y_pred.shape[-1] - y_true.shape[-1]) / 5 + 1)
    c = int(y_true.shape[-1] - (b-1)*5)

    true_conf = K.reshape(y_true_f[:,:,4],(-1,s*s,1,1))
    true_xy = K.expand_dims(y_true_f[:,:,:2],2) # -1,S*S,1,2
    true_wh = K.expand_dims(y_true_f[:,:,2:4],2) # -1,S*S,1,2
    true_class = y_true_f[:,:,5:] # -1,S*S,C

    pred_box = K.reshape(y_pred_f[:,:,:b*5],(-1,s*s,b,5))
    pred_conf = K.expand_dims(pred_box[:,:,:,4],-1) # -1,S*S,B,1
    pred_xy = pred_box[:,:,:,:2] # -1,S*S,B,2
    pred_wh = pred_box[:,:,:,2:4] # -1,S*S,B,2
    pred_class = y_pred_f[:,:,b*5:] # -1,S*S,C

    #responsible 구하기
    true_min_xy = K.minimum(true_xy - true_wh*s/2,true_xy + true_wh*s/2)
    true_max_xy = K.maximum(true_xy - true_wh*s/2,true_xy + true_wh*s/2)

    pred_min_xy = K.minimum(pred_xy - pred_wh*s/2,pred_xy + pred_wh*s/2)
    pred_max_xy = K.maximum(pred_xy - pred_wh*s/2,pred_xy + pred_wh*s/2)

    inter_min_xy = K.maximum(true_min_xy,pred_min_xy)
    inter_max_xy = K.minimum(true_max_xy,pred_max_xy)

    #문제 x

    inter_area = K.maximum(inter_max_xy - inter_min_xy,0.) #max - min 값의 차가 음수가 되면 겹치는 부분이 없다는 것
    true_area = true_max_xy - true_min_xy
    pred_area = pred_max_xy - pred_min_xy

    intersection = inter_area[:,:,:,0] * inter_area[:,:,:,1] 

    
    union = true_area[:,:,:,0]*true_area[:,:,:,1] + pred_area[:,:,:,0]*pred_area[:,:,:,1] - intersection 

    ious = intersection / (union+1e-10) # -1,S*S,B

    ious = K.clip(ious,0.0,1.0)

    best_iou = K.max(ious,-1,True) # -1,S*S,1

    #두 bbox의 iou가 같은 둘 다 responsible이 되는데 이를 방지 같은경우는 하나를 0을 곱해주어 삭제
    same = 1-K.reshape(K.cast(ious[:,:,0]==ious[:,:,1],dtype=ious.dtype),(-1,s*s,1))
    same = K.stack([K.ones_like(best_iou),same],-2)
    
    responsible = K.expand_dims(K.cast(ious >= best_iou,dtype=ious.dtype),-1)*same # -1,S*S,B,1

    localization1 = K.sum(K.square(true_xy-pred_xy)*responsible*true_conf)
    localization2 = K.sum(K.square(K.sqrt(true_wh+1e-10)-K.sqrt(pred_wh+1e-10))*responsible*true_conf)
    conf_obj = K.sum(K.square(1-pred_conf)*responsible*true_conf)
    conf_no_obj = K.sum(K.square(pred_conf)*(1-responsible*true_conf))
    classification = K.sum(K.square(true_class - pred_class)*K.reshape(true_conf,(-1,s*s,1)))

    loss = lambda_coord*(localization1+localization2)+conf_obj+lambda_noobj * conf_no_obj + classification

    return loss 
