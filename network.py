import torch 
import torch.nn as nn 
from torch.nn import functional as  F 
import sys 
from left import LeftModel
from middle import MiddleModule
import thop 
class Model(nn.Module):
    def __init__(self, proj_dim, out_dim, resolution):
        '''
        proj_dim : proj dim in middle model
        out_dim : classes num 
        resolution : resolution of origin image
        '''
        super().__init__()
        self.left_model = LeftModel()
        self.middle_model = MiddleModule()
        self.conv1 = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.BatchNorm2d(proj_dim), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(proj_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)
        )
        self.resolution = resolution
    def forward(self, X):
        tf_output, cnn_output = self.left_model(X) # bs 64 w h, bs 64 wh 
        left_output = [torch.cat((i , j), dim = 1) for (i, j) in zip(tf_output, cnn_output)]
        
        l3, l2, l1 = self.middle_model(left_output)
        l1 = F.upsample(l1, size=l2.size()[-2:], mode="bilinear")
        l2_l1 = self.conv1(l1 + l2)
        l2_l1 = F.upsample(l2_l1, size=l3.size()[-2:],  mode="bilinear")
        l3_l2_l1 = self.conv2(l2_l1 + l3)
        l3_l2_l1 = F.upsample(l3_l2_l1, size=(self.resolution, self.resolution),  mode="bilinear")
        l3_l2_l1 = self.conv3(l3_l2_l1)
        return l3_l2_l1

if __name__ == "__main__":
    device = torch.device("cpu")
    model = Model(64, 5, 128).to(device)
    X = torch.rand(8, 1, 128, 128).to(device)
    # flops, params = thop.profile(model, inputs = (X, ))
    # print(flops / (10 ** 9))
    # print(params)
    print(model(X).shape)