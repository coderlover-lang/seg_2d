import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch 
import numpy as np 
import os 

train_transform_img = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=256, width=256),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(),
    ]
)
test_transform_img = A.Compose(
    [
        ToTensorV2(),
    ]
)


class SegmentationTrianDataset(Dataset):
    def __init__(self, img_path, mode = "train") -> None:
        super().__init__()
        self.img_files = [img_path + i for  i in  os.listdir(img_path)]
        self.mask_files = [i.replace("image", "mask") for i in self.img_files]
        self.mode = mode 
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_file = self.img_files[index]
        mask_file = self.mask_files[index]
        
        img = Image.open(img_file)
        mask = Image.open(mask_file)
        img = np.array(img)
        mask = np.array(mask)
        if self.mode == "train":
            res =  train_transform_img(image = img, mask = mask)
            img = res["image"]
            mask = res["mask"]
        else:
            res =  test_transform_img(image = img, mask = mask)
            img = res["image"]
            mask = res["mask"]
        mask = mask.unsqueeze(0)
        # img = img.unsqueeze(0)
        imgs = [img[:, :128, :128], img[:, :128, 128:256], img[:, 128:256, 128:256], img[:, 128:256, :128]]
        masks = [mask[:, :128, :128], mask[:, :128, 128:256], mask[:, 128:256, 128:256], mask[:, 128:256, :128]]
        # print(imgs[0].shape )
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], dim=0)
        masks = torch.cat([mask.unsqueeze(0) for mask in masks], dim = 0)
        masks = (masks / 63).ceil().long()
        return imgs, masks 
    
if __name__ == "__main__":
    sd  = SegmentationTrianDataset('/home/kathy/projects/project_guo/train/image/', mode="test")
    img, mask = next(iter(sd))
    
    print(img.shape, mask.shape)
    print(torch.unique(mask))
    
