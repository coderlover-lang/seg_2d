import torch
import torch.nn as nn
import torch.nn.functional as F

class SynthesizerAttention(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super(SynthesizerAttention, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Directly parameterize the attention weights
        self.attention_weights = nn.Parameter(torch.randn(seq_len, seq_len))

        # Projection for values
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Directly use parameterized attention weights
        attn_weights = F.softmax(self.attention_weights, dim=-1)  # (seq_len, seq_len)

        # Project values
        v = self.value_proj(x)  # (batch_size, seq_len, embed_dim)

        # Apply attention
        output = torch.einsum('mn,bnd->bmd', attn_weights, v)  # (batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        return output


# Example usage
if __name__ == "__main__":
    sa = SynthesizerAttention(64, 1024)
    input_tensor = torch.randn(4, 1024, 64)  
    output = sa(input_tensor)
    print(output.shape)