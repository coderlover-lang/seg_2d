import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans  # 用于聚类压缩

class ABCAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_clusters=32):
        super(ABCAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_clusters = num_clusters

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Low-rank approximation for attention
        self.low_rank_dim = 64  # 低秩维度
        self.low_rank_proj = nn.Linear(embed_dim, self.low_rank_dim)

    def cluster_compress(self, x):
        """
        使用聚类压缩注意力矩阵。
        """
        batch_size, seq_len, embed_dim = x.size()
        x_flat = x.view(-1, embed_dim).detach().cpu().numpy()  # 将输入展平并转换为 numpy

        # 使用 KMeans 聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(x_flat)  # 聚类标签
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=x.dtype, device=x.device)  # 聚类中心

        # 将输入替换为聚类中心
        x_compressed = cluster_centers[cluster_labels].view(batch_size, seq_len, embed_dim)
        return x_compressed

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Low-rank approximation for queries and keys
        q = self.query_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.key_proj(x)  # (batch_size, seq_len, embed_dim)
        v = self.value_proj(x)  # (batch_size, seq_len, embed_dim)

        # Project queries and keys to low-rank space
        q_low_rank = self.low_rank_proj(q)  # (batch_size, seq_len, low_rank_dim)
        k_low_rank = self.low_rank_proj(k)  # (batch_size, seq_len, low_rank_dim)

        # Compute attention scores in low-rank space
        attn_scores = torch.einsum('bnd,bmd->bnm', q_low_rank, k_low_rank)  # (batch_size, seq_len, seq_len)
        attn_weights = F.softmax(attn_scores / (self.low_rank_dim ** 0.5), dim=-1)

        # Compress values using clustering
        v_compressed = self.cluster_compress(v)  # (batch_size, seq_len, embed_dim)

        # Apply attention
        output = torch.einsum('bnm,bmd->bnd', attn_weights, v_compressed)  # (batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        return output

# Example usage
if __name__ == "__main__":
    batch_size, seq_len, embed_dim = 2, 1024, 64
    abc_attention = ABCAttention(embed_dim, num_heads=4)
    x = torch.randn(batch_size, seq_len, embed_dim)
    output = abc_attention(x)
    print(output.shape)  # Should be (batch_size, seq_len, embed_dim)