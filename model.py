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
    
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        pass

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
        