import torch.nn as nn 
import torch 
from einops import rearrange, einsum
from torch.nn import functional as F 


class OverLapping(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size):
        super().__init__()
        stride = kernel_size // 2
        padding = kernel_size // 4
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channles, kernel_size=kernel_size, stride=stride, padding = padding)
        self.norm = nn.BatchNorm2d(out_channles)

    
    def forward(self, X):
        return rearrange(self.norm(self.conv(X) ), "b c h w -> b (h w) c")
    


if __name__ == "__main__":
    device = torch.device("cuda:3")
    X = torch.rand(16, 3, 128, 128).to(device)
    ol = OverLapping(3, 64, 8).to(device)
    X = ol(X)
    X = rearrange(X, "b (h w) c -> b c h w", h = 32)
    print(X.shape)
    ol = OverLapping(64, 128, 4).to(device)
    X = ol(X)
    print(X.shape)
    
    X = rearrange(X, "b (h w) c -> b c h w", h = 16)
    print(X.shape)
    ol = OverLapping(128, 256, 4).to(device)
    X = ol(X)
    print(X.shape)

    X = rearrange(X, "b (h w) c -> b c h w", h = 8)
    print(X.shape)
    ol = OverLapping(256, 512, 4).to(device)
    X = ol(X)
    print(X.shape)
