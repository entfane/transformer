import torch
import torch.nn as nn

DFF = 2048

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
    
    def __init__(self, vocab_size, embedding_size, num_blocks, num_heads, attn_dim_size, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding = nn.Embedding(seq_len, embedding_size)
        self.blocks = []
        for block in range(num_blocks):
            self.blocks.append(Block(num_heads, attn_dim_size, embedding_size))

    def forward(self, x):
        x = self.embedding(x)
        batch_size, sequence_len, embedding_size = x.shape
        pos_enc = torch.arange(end=sequence_len)
        extended_pos_enc = pos_enc.expand(batch_size, sequence_len)
        extended_pos_enc = self.positional_embedding(extended_pos_enc)
        x = x + extended_pos_enc
        for block in self.blocks:
            x = block(x)

        return x

class Block(nn.Module):

    def __init__(self, num_heads, attn_dim_size, embedding_size):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads, attn_dim_size, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ff = FeedForwardNet(embedding_size)
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
    
class FeedForwardNet(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()
        self.l1 = nn.Linear(embedding_size, DFF)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(DFF, embedding_size)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, attn_dim_size, embedding_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(embedding_size, attn_dim_size, embedding_size // num_heads) for _ in range(num_heads)])
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
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        output = (q @ k.transpose(-2, -1)) / pow(self.attn_dim_size, -1/2)
        output = self.softmax(output)
        output = output @ v
        return output
        