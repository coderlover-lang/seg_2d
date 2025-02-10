import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import os 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from torchvision.transforms import transforms as  T 
from PIL import Image 


map_dic = {
    0.1254902  : 1,  # 右心室
    0.22745098 : 2,  # 左心室
    0.26666668 : 0, 
    0.35686275 : 3,  # 右心房
    0.99215686 : 4, # 左心房
}




class ADatasets(Dataset):
    def __init__(self, mode = "train", return_dir = False) :
        target_files = []
        source_files = []
        start = 0 if mode == "train" else 300
        end = 300 if mode == "train" else -1
        for image_dir in ["/home/kathy/dataset/submit_a/G_image1/", "/home/kathy/dataset/submit_a/R_image1/"]:
            for file in os.listdir(image_dir)[start : end]:
                source_files.append(image_dir + file)
        for image_dir in ["/home/kathy/dataset/submit_a/G_target1/", "/home/kathy/dataset/submit_a/R_target1/"]:
            for file in os.listdir(image_dir)[start : end]:
                target_files.append(image_dir + file)
                
        self.target_files = target_files
        self.source_files = source_files
        
        self.transform = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        self.transform_target = T.Compose([
            T.ToTensor()
        ])
        self.return_dir = return_dir 
        
    def __len__(self):
        return len(self.target_files)
    
    def __getitem__(self, index):
        img = Image.open(self.source_files[index]) 
        target = Image.open(self.target_files[index]) 
        img = self.transform(img)
        target = self.transform_target(target)[[0]] # [1， 800， 600]
        for (k, v) in map_dic.items():
            target[target == k] = v 
        
        imgs = []
        targets = []
        for i in range(6):
            for j in range(4):
                imgs.append(img[None, :, i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128])
                targets.append(target[None, :, i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128])
        imgs = torch.cat(imgs, dim = 0)
        targets = torch.cat(targets, dim = 0)
        if not self.return_dir:
            return imgs, targets
        else:
            return self.source_files[index], imgs, targets
    
def get_dataloader(batch_size = 1):
    train_set = ADatasets(mode="train")
    test_set = ADatasets(mode="test")
    
    return DataLoader(train_set, batch_size=batch_size, shuffle=True) , DataLoader(test_set, batch_size=1, shuffle=False)
    
            
    
         

if __name__ =="__main__":
    
    ad = ADatasets()
    ad = iter(ad)
    while (True):
        X, y = next(ad)
        print(X.shape)
        print(y.shape)