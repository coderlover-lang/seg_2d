import torch.nn as nn 
import torch 
from einops import rearrange, einsum
from torch.nn import functional as F 
import sys 
sys.path.append('/home/kathy/projects/project_guo/test/model')
from attn import EPA_DIM, CustomEPA
from emix_ffn import MIXFFN

class Block(nn.Module):
    def __init__(self, in_features, proj_features, num_heads, down_scale_ratio = 2, resolution = 16):
        super().__init__()
        self.ln_before = nn.LayerNorm(in_features) 
        self.attn_block = CustomEPA(resolution ** 2, proj_features, proj_features, num_heads)
        self.ln_after = nn.LayerNorm(in_features) 
        self.mix_ffn = MIXFFN(resolution, proj_features, mlp_ratio=2)
        self.resolution = resolution
    def forward(self, X):
        '''
        X : [b, 256, 64]
        '''
        X = self.ln_before(X)
        X = self.attn_block(X) 
        X = self.ln_after(X)
        X = self.mix_ffn(X)
        return X 
        
if __name__ == "__main__":
    X = torch.rand(64, 256, 64)
    
    b1= Block(64, 64, 4)
    b2= Block(64, 64, 4)
    X = b1(X)
    X = b2(X)
    print(X.shape)