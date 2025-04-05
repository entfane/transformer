import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        pass
        
    def encode(self, x):
        pass

    def decode(self, x):
        pass


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class Encoder(nn.Module):
    
    def __init__(self, vocab_size, embedding_size, num_blocks, num_heads, attn_dim_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blocks = []
        for block in range(num_blocks):
            self.blocks.append(Block(num_heads, attn_dim_size, embedding_size))

    def forward(self, x):
        pass

class Block(nn.Module):

    def __init__(self, num_heads, attn_dim_size, embedding_size):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads, attn_dim_size, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ff = nn.Linear(embedding_size, embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
    
    def forward(self, x):
        output = self.multi_head_attention(x)
        output += x
        output = self.norm1(output)
        resid = output
        output = self.ff(output)
        output += resid
        output = self.norm2(output)
        return output

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, attn_dim_size, embedding_size):
        super().__init__()
        self.heads = []
        for head in range(num_heads):
            self.heads.append(AttentionHead(embedding_size, attn_dim_size, embedding_size // num_heads))
        self.proj = nn.Linear(embedding_size, embedding_size)
    
    def forward(self, x):
        output = self.heads[0](x)
        for idx in range(1, len(self.heads)):
            output = torch.cat((output, self.heads[idx](x)), dim = -1)
        output = self.proj(output)
        return output
        
        

class AttentionHead(nn.Module):
    
    def __init__(self, embedding_size, attn_dim_size, output_dim_size):
        super().__init__()
        self.attn_dim_size = attn_dim_size
        self.Q = nn.Linear(embedding_size, attn_dim_size)
        self.K = nn.Linear(embedding_size, attn_dim_size)
        self.V = nn.Linear(embedding_size, output_dim_size)
    
    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        output = ((q @ k.T) / pow(self.attn_dim_size, -1/2)) @ v
        return output
        