from custom_dataset.dataset import SegmentationDataset
from custom_dataset.dataugmention import SegmentationTrianDataset
from torch.utils.data import Dataset, DataLoader, random_split



def get_dataloader(mode = "train"):
    
    sd = SegmentationTrianDataset(img_path='/home/kathy/projects/project_guo/train/image/', mode= mode)

    train_set, val_set = random_split(sd, lengths=[len(sd) - len(sd) // 4,  len(sd) // 4])

    train_loader = DataLoader(train_set, 8, shuffle=True)
    val_laoder = DataLoader(val_set, 8, shuffle=False, drop_last=True)

    return train_loader, val_laoder

