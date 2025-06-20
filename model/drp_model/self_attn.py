import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryArrayEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim=128, output_dim=64, num_heads=1, dropout=0.1):
        
        super(BinaryArrayEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(2, embed_dim)
        
        self.pos_encoder = nn.Parameter(torch.zeros(1, input_dim, embed_dim))
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.fc_out = nn.Linear(embed_dim, output_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        embedded = self.embedding(x)  # [batch_size, n*n, embed_dim]
        
        embedded = embedded + self.pos_encoder[:, :embedded.size(1), :]
        
        attn_input = embedded.permute(1, 0, 2)  # [n*n, batch_size, embed_dim]
        
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input)
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, n*n, embed_dim]
        
        attn_output = self.norm1(attn_output + embedded)
        
        pooled = torch.mean(attn_output, dim=1)  # [batch_size, embed_dim]
        
        output = self.fc_out(pooled)  # [batch_size, output_dim]
        output = self.norm2(output)
        
        return output