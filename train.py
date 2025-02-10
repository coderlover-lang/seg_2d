from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch 
import torch.nn as nn 
import pytorch_lightning as pl 
from network import Model
import os 
import random
from einops import rearrange
from torchvision import transforms as  T
from pytorch_lightning.loggers import CSVLogger
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from custom_dataset.adatasets import get_dataloader
from loss.dice import DiceLoss, softmax_dice2
from loss.combination_loss import MultiLoss
from metrics.custom_hd import LabelHd, hd 
import seg_metrics.seg_metrics as sg
import time 

MAX_EPOCHS = 300
LEARNING_RATE = 1e-3



def dice_coef(groundtruth_mask, pred_mask):
    groundtruth_mask = groundtruth_mask[:, 1:, ...]
    pred_mask = pred_mask[:, 1:, ...]
    intersect = torch.sum(pred_mask*groundtruth_mask)
    total_sum = torch.sum(pred_mask) + torch.sum(groundtruth_mask)
    dice = torch.mean(2*intersect/total_sum)
    return dice 

class TrainModel(pl.LightningModule):
    def __init__(self, info = ""):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = Model(64, 5, 128)
        self.loss = nn.CrossEntropyLoss()
        # self.dice = DiceLoss()
        # self.loss = MultiLoss(alpha=0.4)

    def forward(self, X):
        X =  self.model(X)
        return X
    
    
    def training_step(self, batch, batch_idx):
        img, target = batch 
        # img = img / 255
        img =  rearrange(img, "b n c h w -> (b n) c h w")
        target =  rearrange(target, "b n c h w -> (b n) c h w")
        img = img.to(torch.float32)
        target = target.to(torch.int64)
        target = target.squeeze(1)
        
        pred = self(img)
        ce_loss = self.loss(pred, target)
        dice_loss, d1, d2, d3, d4 = softmax_dice2(torch.nn.functional.softmax(pred, dim = 1), target)
        loss = ce_loss + dice_loss
        self.log("train_ce_loss", ce_loss, on_epoch=True, prog_bar=True)
        self.log("d1", d1, on_epoch=True, prog_bar=True)
        self.log("d2", d2, on_epoch=True, prog_bar=True)
        self.log("d3", d3, on_epoch=True, prog_bar=True)
        self.log("d4", d4, on_epoch=True, prog_bar=True)
        self.log("train_dice_loss", dice_loss, on_epoch=True, prog_bar=True)
        
        return loss 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        return [optimizer], [schedular]  
    def test_step(self, batch, batch_idx):
        img, target = batch 
        # img = img / 255
        img =  rearrange(img, "b n c h w -> (b n) c h w")
        target =  rearrange(target, "b n c h w -> (b n) c h w")
        target = target.to(torch.int64)
        img = img.to(torch.float32)
        target = target.squeeze(1)
        mask_true = torch.nn.functional.one_hot(target, 5).permute(0, 3, 1, 2).float()
        start = time.time()
        pred = self(img)
        end = time.time()
        
        dice_loss, dice1, dice2, dice3, dice4 = softmax_dice2(pred.softmax(dim = 1), target)
        self.log("consume", end - start, on_step = True, on_epoch = False)
        # self.log("hd", hd(pred.argmax(dim = 1), target), on_step = True, on_epoch = False)
        self.log("hd", hd95( target, pred), on_step = True, on_epoch = False)
        self.log("dice1", dice1, on_step = True, on_epoch = False)
        self.log("dice2", dice2, on_step = True, on_epoch = False)
        self.log("dice3", dice3, on_step = True, on_epoch = False)
        self.log("dice4", dice4, on_step = True, on_epoch = False)
        self.log("dice", dice_coef( mask_true, pred.softmax(dim = 1)) , on_step = True, on_epoch = False)
        

def hd95(gdth_img, pred_img):
    # gdth : 24,  128, 128 
    # pred : 24, 5, 128, 128
    total_hd1, total_hd2, total_hd3, total_hd4 = 0 , 0, 0, 0
    gdth_img = gdth_img.detach().cpu().numpy()
    pred_img = pred_img.detach().cpu().numpy().argmax(axis = 1)
    for i in range(24):
        gt_temp = gdth_img[i, ...]
        pr_temp = pred_img[i, ...]
        gt_temp[0, 0], pr_temp[0, 0] = 4, 4
        gt_temp[1, 1], pr_temp[1, 1] = 1, 1
        gt_temp[2, 2], pr_temp[2, 2] = 2, 2
        gt_temp[3, 3], pr_temp[3, 3] = 3, 3
        hd1, hd2, hd3, hd4 =   sg.write_metrics(labels=[1, 2, 3, 4], gdth_img=gt_temp, pred_img=pr_temp, metrics=["hd95"])[0]["hd95"]
        total_hd1 = total_hd1 + hd1 
        total_hd2 = total_hd2 + hd2 
        total_hd3 = total_hd3 + hd3 
        total_hd4 = total_hd4 + hd4 
    return (total_hd1 + total_hd2 + total_hd3 + total_hd4) / 4 /24
if __name__ == "__main__":
    mode = "train"
    check_point_path = '/home/kathy/projects/project_guo/test/logs/train_a/version_2/checkpoints/epoch=67-step=30532.ckpt'
    train_loader, test_laoder = get_dataloader(batch_size=2)
    if mode == "train":
        info = '''仅CE loss '''

        model = TrainModel(info)
        
        logger = CSVLogger("logs", name="sub_train")
        trainer = pl.Trainer(accelerator="gpu", devices=[1], precision=32, max_epochs=MAX_EPOCHS, logger=logger)
        trainer.fit(model, train_loader)
    elif mode == "test":
        logger = CSVLogger("logs", name="test_a")
        model = TrainModel.load_from_checkpoint(checkpoint_path=check_point_path)
        trainer = pl.Trainer(accelerator="gpu", devices=[1], precision=32, max_epochs=MAX_EPOCHS, logger=logger)
        
        trainer.test(model, test_laoder)
    else:
        gdth_img = torch.randint(0, 4, size = (24, 128, 128))
        pred_img = torch.randint(0, 2, size = (24, 5, 128, 128))
        print(hd95(gdth_img, pred_img))
    
    