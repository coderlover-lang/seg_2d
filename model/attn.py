import torch.nn as nn 
import torch 
from einops import rearrange, einsum
from torch.nn import functional as F 
import thop 

class EPA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices used in spatial attention module to project keys and values from HWD-dimension to P-dimension
        self.E = nn.Linear(input_size, proj_size)
        self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape
        #print("The shape in EPA ", self.E.shape)

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}


    
class EPA_DIM(nn.Module):
    def __init__(self, in_features, proj_features, num_heads, down_scale_ratio = 2):
        super().__init__()
        self.to_qkvv = nn.Linear(in_features, proj_features * 4)
        self.num_heads = num_heads
        self.down_sample_k = nn.Conv2d(in_features // num_heads, proj_features // num_heads, kernel_size=down_scale_ratio, stride=down_scale_ratio)
        self.down_sample_v = nn.Conv2d(in_features // num_heads, proj_features // num_heads, kernel_size=down_scale_ratio, stride=down_scale_ratio)
    def forward(self, X):
        B, N, C = X.shape
        ws = torch.sqrt(torch.tensor(N)).type_as(X).to(torch.long)
        
        q_shared, k_shared, v1, v2 = self.to_qkvv(X).chunk(4, dim = 1) # B, N, C
        q_shared, k_shared, v1, v2 = map(
            lambda x : rearrange(X, "B N (nh hd) -> B nh N hd", nh = self.num_heads), 
            (q_shared, k_shared, v1, v2), 
        )
        attns_channle = einsum(q_shared, k_shared, "B nh i hd, B nh j hd -> B nh i j")
        attns_channle = torch.sigmoid(attns_channle)
        channle_output = einsum(attns_channle, v2, "B nh i j, B nh j hd -> B nh i hd")
        
        k_shared = rearrange(k_shared, "B nh (h w) hd -> (B nh) hd h w", h = ws)
        k_shared_down = self.down_sample_k(k_shared)
        k_shared_down = rearrange(k_shared_down, "(b nh) p h w -> b nh (h w) p", nh = self.num_heads)
        attns_spatial  = einsum(q_shared, k_shared_down, "B nh i hd, B nh j hd -> B nh i j")
        
        v1 = rearrange(v1, "B nh (h w) hd -> (B nh) hd h w", h = ws)
        v1 = self.down_sample_v(v1)
        v1 = rearrange(v1, "(b nh) p h w -> b nh (h w) p", nh = self.num_heads)
        spatial_output = einsum(attns_spatial, v1, "B nh i j, B nh j hd -> B nh i hd")
        
        spatial_output = rearrange(spatial_output, "b nh N hd -> b N (nh hd)")
        channle_output = rearrange(channle_output, "b nh N hd -> b N (nh hd)")
        
        return spatial_output + channle_output + X
        
        
# class EPA_DIM2(nn.Module):
#     def __init__(self, in_features, proj_features, input_size,proj_size,num_heads, down_scale_ratio = 2):
#         super().__init__()
#         self.to_qkvv = nn.Linear(in_features, proj_features * 4)
#         self.num_heads = num_heads
#         self.down_sample_k = nn.Conv2d(in_features // num_heads, proj_features // num_heads, kernel_size=down_scale_ratio, stride=down_scale_ratio)
#         self.down_sample_v = nn.Conv2d(in_features // num_heads, proj_features // num_heads, kernel_size=down_scale_ratio, stride=down_scale_ratio)
#         self.E = nn.Linear(input_size, proj_size)
#         self.F = nn.Linear(input_size, proj_size)
#     def forward(self, X):
#         B, N, C = X.shape
#         ws = torch.sqrt(torch.tensor(N)).type_as(X).to(torch.long)
        
#         q_shared, k_shared, v1, v2 = self.to_qkvv(X).chunk(4, dim = 1) # B, N, C
#         q_shared, k_shared, v1, v2 = map(
#             lambda x : rearrange(X, "B N (nh hd) -> B nh N hd", nh = self.num_heads), 
#             (q_shared, k_shared, v1, v2), 
#         )
#         attns_channle = einsum(q_shared, k_shared, "B nh i hd, B nh j hd -> B nh i j")
#         attns_channle = torch.sigmoid(attns_channle)
#         channle_output = einsum(attns_channle, v2, "B nh i j, B nh j hd -> B nh i hd")
        
#         k_shared = rearrange(k_shared, "B nh (h w) hd -> (B nh) hd h w", h = ws)
#         k_shared_down = self.down_sample_k(k_shared)
#         k_shared_down = rearrange(k_shared_down, "(b nh) p h w -> b nh (h w) p", nh = self.num_heads)
#         attns_spatial  = einsum(q_shared, k_shared_down, "B nh i hd, B nh j hd -> B nh i j")
        
#         v1 = rearrange(v1, "B nh (h w) hd -> (B nh) hd h w", h = ws)
#         v1 = self.down_sample_v(v1)
#         v1 = rearrange(v1, "(b nh) p h w -> b nh (h w) p", nh = self.num_heads)
#         spatial_output = einsum(attns_spatial, v1, "B nh i j, B nh j hd -> B nh i hd")
        
#         spatial_output = rearrange(spatial_output, "b nh N hd -> b N (nh hd)")
#         channle_output = rearrange(channle_output, "b nh N hd -> b N (nh hd)")
        
#         return spatial_output + channle_output + X

class CustomEPA(nn.Module):
    def __init__(self  ,input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)
         
        self.E = nn.Linear(input_size, proj_size)
        self.F = nn.Linear(input_size, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))
    
    def forward(self, x):
        B, N, C = x.shape
        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]
        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

        

if __name__ == "__main__":
    num_heads = 4
    attn = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8)
    X = torch.rand(8, 32 ** 2, 64)
    epa = CustomEPA(32 ** 2 , 64, 64 )
    
    flops, parmas = thop.profile(epa, inputs=(X, ))
    print(flops / (10 ** 9))
    print(parmas)