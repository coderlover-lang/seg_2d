import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import os 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from torchvision.transforms import transforms as  T 


class KitsDataset(Dataset):
    def __init__(self):
        
        self.current_folder_index = 0
        # self.folder_ 
        self.seg_base_dir = "/home/kathy/dataset/LITS/clean/"
        self.origin_base_dir = "/home/kathy/dataset/LITS/origin_clean/"
        self.seg_files, self.origin_files = [], []
        for file in os.listdir(self.seg_base_dir):
            for sub_file in os.listdir(self.seg_base_dir + file):
                self.seg_files.append(self.seg_base_dir + file + "/" + sub_file)
                self.origin_files.append(self.origin_base_dir + file + "/" + sub_file)
        
        print("data init")
        
        
        
        
    
    def __len__(self):
        return len(self.seg_files)
    
    def __getitem__(self, index):
        seg = np.load(self.seg_files[index])
        origin = np.load(self.origin_files[index])
        segs = []
        origins = []
        for i in range(4):
            for j in range(4):
                segs.append(seg[None, i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128])
                origins.append(origin[None, i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128])
        
        # print([segs[i].shape for i in range(16)])
        seg = torch.from_numpy(np.concatenate(segs, axis = 0))
        origin = torch.from_numpy(np.concatenate(origins, axis = 0))
        origin = (origin - origin.min()) / (origin.max() - origin.min())
        
        
        return seg.unsqueeze(1), origin.unsqueeze(1)
        
def get_dataloader(batch_size = 1):
    return DataLoader(KitsDataset(), batch_size=batch_size, shuffle=True)
    
            
         
if __name__ == "__main__":
    kd = KitsDataset()
    seg, origin = next(iter(kd))
    print(origin.min(), origin.max())
    print(seg.shape)
    print(origin.shape)
    