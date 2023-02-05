import keras.backend as K

def loss_function(y_true,y_pred):
    #y_true,y_pred 모두 ([B,W,H,C])로 들어온다.
    #이때 y_true 는 (s,s,5+c)로 들어오고 y pred는 (s,s,b*5+c)로 들어온다.
    #x,y,w,h,confidence

    lambda_coord = 5
    lambda_noobj = 0.5

    batch_size = y_true.shape[0]
    s = y_true.shape[1]
    b = int((y_pred.shape[-1] - y_true.shape[-1]) / 5 + 1)
    c = int(y_true.shape[-1] - (b-1)*5)
    print(s,b,c)

    y_true_f = K.reshape(y_true,(-1,s*s,5+c))
    y_pred_f = K.reshape(y_pred,(-1,s*s,b*5+c))
    loss = []
    for i in range(batch_size):

        cell_loss = []
        for j in range(s*s):
            is_exist = y_true_f[i,j,4]
            # Responsible 구하기
            iou_list = []
            for k in range(b):
                xt,yt,wt,ht,ct = y_true_f[i,j,0:5]
                xp,yp,wp,hp,cp = y_pred_f[i,j,k*5:k*5+5]
                
                x1t,y1t,x2t,y2t = [xt-0.5*wt,yt-0.5*ht,xt+0.5*wt,yt+0.5*ht]
                x1p,y1p,x2p,y2p = [xp-0.5*wp,yp-0.5*hp,xp+0.5*wp,yp+0.5*hp]
                
                x1i,y1i,x2i,y2i = [max(x1t,x1p),max(y1t,y1p),min(x2t,x2p),min(y2t,y2p)]

                intersection = 0
                if x2i - x1i >= 0 and y2i - y1i >= 0:
                    intersection = (x2i-x1i) * (y2i-y1i) 


                union = (x2t-x1t) * (y2t-y1t) + (x2p-x1p) * (y2p-y1p) - intersection

                if union == 0:
                    iou_list.append(0)
                else :
                    iou_list.append(intersection/union)

            responsible = K.argmax(iou_list)*5

            localization1 = K.pow(y_true_f[i,j,0]-y_pred_f[i,j,responsible],2) + K.pow(y_true_f[i,j,1]-y_pred_f[i,j,responsible+1],2)
            localization2 = K.pow(K.sqrt(y_true_f[i,j,2])-K.sqrt(y_pred_f[i,j,responsible+2]),2) + K.pow(K.sqrt(y_true_f[i,j,3])-K.sqrt(y_pred_f[i,j,responsible+3]),2)

            localization = lambda_coord * (localization1+localization2)

            
            confidence = 0
            if is_exist:
                #객체가 있는 경우에는 responsible만 이용해 loss구하기
                confidence = K.pow(y_true_f[i,j,4]-y_pred_f[i,j,responsible+4],2)
            else :
                #객체가 없는 경우에는 Bounding box를 이용해 loss구하기
                for k in range(b):
                    confidence += K.pow(y_true_f[i,j,4]-y_pred_f[i,j,k*5+4],2)
                confidence *= lambda_noobj
            
            classification = 0
            for k in range(c):
                classification+= K.pow(y_true_f[i,j,5+k] - y_pred_f[i,j,b*5+k],2)
            

            cell_loss.append(localization+confidence+classification)
        loss.append(sum(cell_loss)/s**2)
    ret = sum(loss)/batch_size
    return ret