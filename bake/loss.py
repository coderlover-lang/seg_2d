def substract(x,y):
    """
    求2个2维0-1矩阵相减，用于求FP,FN
    :return:value
    """
    z=x-y
    z=np.maximum(z,0)
    return np.sum(z)
def substract_torch(x,y):
    """
    求2个2维0-1矩阵相减，用于求FP,FN
    :return:value
    """
    z=x-y
    z=torch.maximum(z,torch.tensor(0))
    return torch.sum(z)

def sen_ppv(mask_pred:torch.tensor,mask_true:torch.tensor,epsion=1e-4):
    #仅适用于二维
    mp,mt=mask_pred.to('cpu').numpy(),mask_true.to('cpu').numpy()
    mp=mp[:, 1:, ...]
    mt=mt[:, 1:, ...]
    dice_res,sen_res,ppv_res=0,0,0
    bs=mp.shape[0]
    for i in range(bs):
        mtt=np.squeeze(mt[i])
        mpp=np.squeeze(mp[i])
        TP=np.sum(mtt*mpp)
        FP=substract(mpp,mtt)
        FN=substract(mtt,mpp)
        dice_res+=2*TP/(2*TP+FP+FN+epsion)
        sen_res+=TP/(TP+FN+epsion)
        ppv_res+=TP/(TP+FP+epsion)
    return dice_res/bs,sen_res/bs,ppv_res/bs

def sen_ppv_torch(mask_pred:torch.tensor,mask_true:torch.tensor,epsion=torch.tensor(1e-4)):
    #仅适用于二维
    mp=mask_pred[:, 1:, ...]
    mt=mask_true[:, 1:, ...]
    dice_res,sen_res,ppv_res=torch.tensor(0).type_as(mask_pred).float(), torch.tensor(0).type_as(mask_pred).float(), torch.tensor(0).type_as(mask_pred).float()
    bs=mp.shape[0]
    for i in range(bs):
        mtt=torch.squeeze(mt[i])
        mpp=torch.squeeze(mp[i])
        TP=torch.sum(mtt*mpp)
        FP=substract_torch(mpp,mtt)
        FN=substract_torch(mtt,mpp)
        dice_res+=2*TP/(2*TP+FP+FN+epsion)
        sen_res+=TP/(TP+FN+epsion)
        ppv_res+=TP/(TP+FP+epsion)
    return dice_res/bs,sen_res/bs,ppv_res/bs