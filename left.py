import torch.nn as nn 
import torch 
from einops import rearrange, einsum
from torch.nn import functional as F 
from model.patch_merging import OverLapping
from model.adjust_transformer_block import Block as MixBlock
from model.resnet34 import Resnet
class StageModuleTfBlock(nn.Module):
    def __init__(self, in_channels, out_channles, kernel_size, depth, down_resolution, num_heads, down_scale_ratio):
        '''
        in_channles     : channle of tensor 
        in_channles     : channle of output 
        kernel_size     : kernel_size of conv2d 
        resolution      : W or H of Input image 
        down_resolution : W or H of Output
        num_heads       : The number of heads of attn block 
        down_scale_ratio: p ratio of EPA 
        '''
        super().__init__()
        self.patch_merging = OverLapping(in_channels, out_channles, kernel_size)
        self.mix_block = nn.Sequential(
            *[MixBlock(out_channles, out_channles, num_heads=num_heads, down_scale_ratio=down_scale_ratio, resolution=down_resolution) for i in range(depth)]
        )
    def forward(self, X):
        X = self.patch_merging(X)
        X =  self.mix_block(X)
        
        return X


class StageModuleTf(nn.Module):
    def __init__(
        self, 
        in_channels = [1, 64, 128, 256], 
        out_channels = [64, 128, 256, 512], 
        kernel_sizes = [8, 4, 4, 4], 
        depths = [2, 6, 2, 2], 
        down_resolutions = [32, 16, 8, 4],
        num_heads = [4, 4, 4, 4]
    ):
        super().__init__()
        for i in range(len(in_channels)):
            setattr(self, "tf_block_" + str(i),
                StageModuleTfBlock(
                    in_channels=in_channels[i], 
                    out_channles=out_channels[i], 
                    kernel_size=kernel_sizes[i], 
                    depth=depths[i], 
                    down_resolution=down_resolutions[i], 
                    num_heads=num_heads[i], 
                    down_scale_ratio=2
                )
            )
        self.total_stage = len(depths)
        self.resolutions = down_resolutions
    def forward(self, X):
        tf_output = []
        for  i in range(self.total_stage):
            X = getattr(self, "tf_block_" + str(i))(X)
            X = rearrange(X, "b (h w) c -> b c h w", h = self.resolutions[i])
            tf_output.append(X)
        return tf_output
    
class LeftModel(nn.Module):
    def __init__(
        self, 
        in_channels = [1, 64, 128, 256], 
        out_channels = [64, 128, 256, 512], 
        kernel_sizes = [8, 4, 4, 4], 
        depths = [2, 6, 2, 2], 
        down_resolutions = [32, 16, 8, 4],
        num_heads = [4, 4, 4, 4]
    ):
        super().__init__()
        self.tf_model = StageModuleTf(in_channels, out_channels, kernel_sizes, depths, down_resolutions, num_heads)
        self.cnn_model = Resnet(kernel_sizes, in_channels, out_channels, [3, 4, 6, 3])
    
    def forward(self, X):
        tf_output = self.tf_model(X)
        cnn_output = self.cnn_model(X)
        return tf_output, cnn_output
if __name__ == "__main__":
    device = torch.device("cuda:3")
    device = torch.device("cpu")
    X = torch.rand(16, 1, 128, 128).to(device)
    # ol = StageModuleTfBlock(1, 64, 8, 2, 32, 4, 2).to(device)
    # X = ol(X)
    # X = rearrange(X, "b (h w) c -> b c h w", h = 32)
    # print(X.shape)
    # ol = StageModuleTfBlock(64, 128, 8, 2, 32, 4, 2).to(device)
    model = LeftModel().to(device)
    tf_output, cnn_output = model(X)
    print([i.shape for i in tf_output])
    print([i.shape for i in cnn_output])
    
    