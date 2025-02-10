import torch.nn as nn 
import torch 
from einops import rearrange, einsum
from torch.nn import functional as F 



class MIX_BLOCK(nn.Module):
    def __init__(self, in_features, resolution):
        super().__init__()
        self.conv = nn.Conv2d(in_features, in_features, kernel_size=1)
        self.ln1 = nn.LayerNorm(in_features)
        self.ln2 = nn.LayerNorm(in_features)
        self.ln3 = nn.LayerNorm(in_features)
        self.resolution = resolution
    def forward(self, X):
        identity = X 
        X = rearrange(X, "b (h w) c -> b c h w", h = self.resolution)
        X = self.conv(X)
        X = rearrange(X, "b c h w -> b (h w) c")
        X = identity + X 
        X = self.ln1(X)
        X = X + identity
        X = self.ln2(X)
        X = X + identity
        X = self.ln3(X)
        X = X + identity

        return X 
class MIXFFN(nn.Module):
    def __init__(self, resolution, in_features, mlp_ratio):
        super().__init__()
        self.lp1 = nn.Linear(in_features, in_features * mlp_ratio)
        self.mix_block = MIX_BLOCK(in_features * mlp_ratio, resolution)
        self.gelu = nn.GELU()
        self.lp2 = nn.Linear(in_features * mlp_ratio , in_features)
        self.resolution = resolution
    def forward(self, X):
        X =  self.lp1(X)
        X = self.mix_block(X)
        X = self.gelu(X)
        return self.lp2(X)
    
if __name__ == "__main__":
    mix_ffn = MIXFFN(16, 64, 4)
    X = torch.rand(64, 256, 64)
    X = mix_ffn(X)
    print(X.shape)
        
    