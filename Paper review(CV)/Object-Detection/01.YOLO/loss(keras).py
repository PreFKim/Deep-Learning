import keras.backend as K

def loss_function(y_true,y_pred,lambda_coord=5,lambda_noobj=0.5):
    #이때 y_true 는 (s,s,5+c)로 들어오고 y pred는 (s,s,b*5+c)로 들어온다.
    #(x,y,w,h,confidence)*b,classes(c)

    batch_size = y_true.shape[0]
    s = y_true.shape[1]
    b = (y_pred.shape[-1] - y_true.shape[-1]) // 5 + 1
    c = y_true.shape[-1] - (b-1)*5

    y_true_f = K.reshape(y_true,(-1,s*s,5+c))
    y_pred_f = K.reshape(y_pred,(-1,s*s,b*5+c))

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
    true_min_xy = true_xy-true_wh/2
    true_max_xy = true_xy+true_wh/2

    pred_min_xy = pred_xy-pred_wh/2
    pred_max_xy = pred_xy+pred_wh/2


    inter_min_xy = K.maximum(true_min_xy,pred_min_xy)
    inter_max_xy = K.minimum(true_max_xy,pred_max_xy)

    inter_area = K.maximum(inter_max_xy-inter_min_xy,0)
    true_area = true_max_xy - true_min_xy
    pred_area = pred_max_xy - pred_min_xy

    intersection = inter_area[:,:,:,0] * inter_area[:,:,:,1] 
    
    union = true_area[:,:,:,0]*true_area[:,:,:,1] + pred_area[:,:,:,0]*pred_area[:,:,:,1] - intersection

    ious = intersection / union # -1,S*S,B

    best_iou = K.max(ious,-1,True) # -1,S*S,1

    responsible = K.expand_dims(K.cast(ious >= best_iou,dtype=ious.dtype),-1) # -1,S*S,B,1

    #최종 loss 구하기
    localization1 = K.sum(K.pow(true_xy-pred_xy,2)*responsible*true_conf) 
    localization2 = K.sum(K.pow(K.sqrt(true_xy)-K.sqrt(pred_xy),2)*responsible*true_conf)
    conf_obj = K.sum(K.pow(true_conf-pred_conf,2)*responsible*true_conf)
    conf_no_obj = K.sum(K.pow(true_conf-pred_conf,2)*(1-responsible*true_conf))
    classification = K.sum(K.pow(pred_class - true_class,2)*K.reshape(true_conf,(-1,s*s,1)))

    loss = lambda_coord*(localization1+localization2)+conf_obj+lambda_noobj * conf_no_obj + classification

    return loss 
