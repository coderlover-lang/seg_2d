import torch.nn as nn 
import torch 
from loss.dice import DiceLoss




class MultiLoss(nn.Module):
    def __init__(self, alpha = .5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.alpha = alpha 
    def forward(self, X, y):
        ce_loss = self.ce_loss(X, y)
        dice_loss, dice1, dice2, dice3, dice4 = self.dice_loss(X, y)
        
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss, dice1, dice2, dice3, dice4 
    

if __name__ == "__main__":
    out = torch.rand(2,5,128,128)
    tar = torch.randint(0, 4,  size=(2,128,128))
    ml = MultiLoss()
    print(ml(out,tar))
        