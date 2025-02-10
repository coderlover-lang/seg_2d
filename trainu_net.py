from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch 
import torch.nn as nn 
import pytorch_lightning as pl 
from network import Model
from torch.utils.data import Dataset, DataLoader, random_split
import os 
from PIL import Image
import random
from einops import rearrange
from torchvision import transforms as  T
from pytorch_lightning.loggers import CSVLogger
import numpy as np
from Unet.unet import UNet
from custom_dataset.dataset import get_dataloader
from loss.dice import softmax_dice2
MAX_EPOCHS = 100

def substract(x,y):
    """
    求2个2维0-1矩阵相减，用于求FP,FN
    :return:value
    """
    z=x-y
    z=np.maximum(z,0)
    return np.sum(z)

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


class TrainModel(pl.LightningModule):
    def __init__(self, info = ""):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(1, 5)
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):
        X =  self.model(X)
        return X 
    
    def training_step(self, batch, batch_idx):
        img, target = batch 
        # img = img / 255
        img =  rearrange(img, "b n c h w -> (b n) c h w")
        target =  rearrange(target, "b n c h w -> (b n) c h w")
        target = target.squeeze(1)

        pred = self(img)
        ce_loss = self.loss(pred, target)
        dice_loss, dice1, dice2, dice3, dice4 = softmax_dice2(pred, target)
        
        self.log("train_ce_loss", ce_loss, on_epoch=True, prog_bar=True)
        self.log("train_dice_loss", dice_loss, on_epoch=True, prog_bar=True)
        self.log("dice1", dice1, on_epoch=True, prog_bar=True)
        
        return (0.5 * ce_loss + 0.5 * dice_loss) 

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        return [optimizer], [schedular]  
    def test_step(self, batch, batch_idx):
        img, target = batch 
        # img = img / 255
        img =  rearrange(img, "b n c h w -> (b n) c h w")
        target =  rearrange(target, "b n c h w -> (b n) c h w")
        target = target.squeeze(1)
        mask_true = torch.nn.functional.one_hot(target, 5).permute(0, 3, 1, 2).float()
        
        pred = self(img)
        
        dice_loss, dice1, dice2, dice3, dice4 = softmax_dice2(pred, target)
        self.log("dice1", dice1)
        self.log("dice2", dice2)
        self.log("dice3", dice3)
        self.log("dice4", dice4)
        
        self.log("dice", sen_ppv(pred, mask_true)[0])
        
if __name__ == "__main__":
    mode = "train"
    chech_pointpath = '/home/kathy/projects/project_guo/test/logs/logger_unet/version_1/checkpoints/epoch=14-step=915.ckpt'
    train_loader, val_laoder = get_dataloader()
    if mode == "train":
        info = "Train Unet"
        model = TrainModel(info)
            
        logger = CSVLogger("logs", name="logger_unet")
        trainer = pl.Trainer(accelerator="gpu", devices=[8], precision=16, max_epochs=MAX_EPOCHS, logger=logger)
        trainer.fit(model, train_loader)
    elif mode == "test":
        model = TrainModel.load_from_checkpoint(checkpoint_path = chech_pointpath)
        logger = CSVLogger("logs", name="logger_unet_test")
        trainer = pl.Trainer(accelerator="gpu", devices=[8], precision=16, max_epochs=MAX_EPOCHS, logger=logger)
        
        trainer.test(model, val_laoder)
        