import torch 
from torch import nn as nn 



def Dice(output, target, eps=torch.tensor(1e-5)):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den


def softmax_dice2(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 3).float())
    loss4 = Dice(output[:, 4, ...], (target == 4).float())

    return loss1 + loss2 + loss3 + loss4, 1-loss1.data, 1-loss2.data, 1-loss3.data, 1 - loss4.data 


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, X, y):
        return softmax_dice2(X, y) 



if __name__ == "__main__":
    out = torch.rand(2,5,128,128)
    tar = torch.rand(2,128,128)
    softmax_dice2(out,tar)