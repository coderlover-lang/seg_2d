import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math 
class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, n, c = x.size()
        height, width = int(math.sqrt(n)), int(math.sqrt(n))
        x = rearrange(x, "b (h w) c -> b c h w", h = height, w = width)
        
        # Compute query, key, value
        proj_query = self.query_conv(x).view(bs, -1, width*height).permute(0, 2, 1) # BxNxC'
        proj_key = self.key_conv(x).view(bs, -1, width*height) # BxC'xN
        proj_value = self.value_conv(x).view(bs, -1, width*height) # BxCxN
        
        # Criss-cross attention
        energy = torch.bmm(proj_query, proj_key) # BxNxN
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # BxCxN
        out = out.view(bs, -1, height, width)

        out = self.gamma*out + x
        return rearrange(out, "b c h w -> b (h w) c")
    


class MultiHeadCrissCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(MultiHeadCrissCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * num_heads, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * num_heads, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * num_heads, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels=in_channels * num_heads, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, n, c = x.size()
        height, width = int(math.sqrt(n)), int(math.sqrt(n))
        x = rearrange(x, "b (h w) c -> b c h w", h = height, w = width)
        
        # Compute query, key, value for each head
        proj_query = self.query_conv(x).view(bs, self.num_heads, -1, width*height).permute(0, 1, 3, 2) # BxHxNx(C'/H)
        proj_key = self.key_conv(x).view(bs, self.num_heads, -1, width*height) # BxHx(C'/H)xN
        proj_value = self.value_conv(x).view(bs, self.num_heads, -1, width*height) # BxHxCxN
        
        # Criss-cross attention for each head
        energy = torch.matmul(proj_query, proj_key) # BxHxNxN
        attention = F.softmax(energy, dim=-1)
        out = torch.matmul(proj_value, attention.permute(0, 1, 3, 2)) # BxHxCxN
        
        # Concatenate outputs from all heads and apply final linear layer
        out = out.view(bs, -1, height, width)
        out = self.final_conv(out)
        
        out = self.gamma*out + x
        return rearrange(out, "b c h w -> b (h w) c")




if __name__ == "__main__":
    ccnet = MultiHeadCrissCrossAttention(in_channels=64, out_channels=128)
    input_tensor = torch.randn(4, 1024, 64)  
    output_tensor = ccnet(input_tensor)
    print(output_tensor.shape) 