import torch 
from torchvision import transforms as T 
from torch.utils.data import Dataset, DataLoader, random_split
import os 
from PIL import Image
import random
from albumentations.pytorch import ToTensorV2
import json 

class SegmentationDataset(Dataset):
    def __init__(self, img_path, mode = "train") -> None:
        super().__init__()
        with open('/home/kathy/projects/project_guo/train_info.json', 'r', encoding="utf8") as f:
            dic  = json.loads(f.read())
        
        self.img_files = [img_path + i for  i in  os.listdir(img_path)]
        self.img_files = [self.img_files[i] for i in dic[mode]]
        self.mask_files = [i.replace("image", "mask") for i in self.img_files]
        self.transforms_img = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean=[0.5], std=[0.5])
            
        ])
        self.transforms_mask = T.Compose([
            T.ToTensor()
        ])
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        
        img = Image.open(img_file)
        mask = Image.open(mask_file)
        img = self.transforms_img(img)
        mask = self.transforms_mask(mask)
        
        imgs = [img[:, :128, :128], img[:, :128, 128:256], img[:, 128:256, 128:256], img[:, 128:256, :128]]
        masks = [mask[:, :128, :128], mask[:, :128, 128:256], mask[:, 128:256, 128:256], mask[:, 128:256, :128]]
        
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], dim=0)
        masks = torch.cat([mask.unsqueeze(0) for mask in masks], dim = 0)
        masks = (masks * 4).ceil().long()
        return imgs, masks 
    

def get_dataloader(batch_size = 1):
    
    
    train_sd = SegmentationDataset(img_path='/home/kathy/projects/project_guo/train/image/', mode="train")
    test_sd = SegmentationDataset(img_path='/home/kathy/projects/project_guo/train/image/', mode="test")


    train_loader = DataLoader(train_sd, batch_size=batch_size , shuffle=True)
    val_laoder = DataLoader(test_sd, batch_size=batch_size , shuffle=False)
    
    return train_loader, val_laoder


def predict_by_name_processing(image_name):
    img_file = "/home/kathy/projects/project_guo/train/image/" + image_name
    mask_file = "/home/kathy/projects/project_guo/train/mask/" + image_name
    transforms_img = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean=[0.5], std=[0.5])
            
        ])
    transforms_mask = T.Compose([
        T.ToTensor()
    ])
    img = Image.open(img_file)
    mask = Image.open(mask_file)
    img = transforms_img(img)
    mask = transforms_mask(mask)
    imgs = [img[:, :128, :128], img[:, :128, 128:256], img[:, 128:256, 128:256], img[:, 128:256, :128]]
    masks = [mask[:, :128, :128], mask[:, :128, 128:256], mask[:, 128:256, 128:256], mask[:, 128:256, :128]]
    imgs = torch.cat([img.unsqueeze(0) for img in imgs], dim=0)
    masks = torch.cat([mask.unsqueeze(0) for mask in masks], dim = 0)
    masks = (masks * 4).ceil().long()
    return imgs, masks.squeeze(1)



if __name__ == "__main__":
    # sd = SegmentationDataset(img_path='/home/kathy/projects/project_guo/train/image/')
    # sd = SegmentationDataset(img_path='/home/kathy/projects/project_guo/train/image/',mode='test')
    # print(sd.mask_files)
    tra, val = get_dataloader()
    imgs, masks = next(iter(tra))
    print(imgs.shape, masks.shape)
    # imgs, masks = predict_by_name_processing("15_18.png")
    # print(imgs.shape)
    # print(masks.shape)
        
    
    