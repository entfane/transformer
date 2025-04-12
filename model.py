import torch
import torch.nn as nn

DFF = 2048
SEQ_LEN = 8
EMBEDDING_SIZE = 32
VOCAB_SIZE = 64
NUM_BLOCKS = 8
NUM_HEADS = 8
ATTN_DIM_SIZE = EMBEDDING_SIZE // NUM_HEADS

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.pos_embedding = nn.Embedding(SEQ_LEN, EMBEDDING_SIZE)
        self.blocks = nn.ModuleList([Block() for i in range(NUM_BLOCKS)])
        self.l = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        input = self.embedding(x)
        input_pos = self.pos_embedding(torch.arange(SEQ_LEN))
        input += input_pos
        for block in self.blocks:
            input = block(input)
        input = self.l(input)
        input = self.softmax(input)
        return input

class Block(nn.Module): 

    def __init__(self):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(EMBEDDING_SIZE)
        self.multi_head_attention = MultiHeadAttention()
        self.norm2 = nn.LayerNorm(EMBEDDING_SIZE)
        self.ff = FeedForwardNet()
        self.norm3 = nn.LayerNorm(EMBEDDING_SIZE)
        self.register_buffer('tril', torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)))
    
    def forward(self, x):
        output = self.masked_multi_head_attention(x, self.tril)
        output += x
        output = self.norm1(output)
        resid = output

        output = self.multi_head_attention(output)
        output += resid
        output = self.norm2(output)
        resid = output

        output = self.ff(output)
        output += resid
        output = self.norm3(output)
        return output
    
class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(EMBEDDING_SIZE, DFF)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(DFF, EMBEDDING_SIZE)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class MultiHeadAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead() for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
    
    def forward(self, x, mask = None):
        output = self.heads[0](x)
        for idx in range(1, len(self.heads)):
            output = torch.cat((output, self.heads[idx](x, mask)), dim = -1)
        output = self.proj(output)
        return output
        
        

class AttentionHead(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(EMBEDDING_SIZE, ATTN_DIM_SIZE)
        self.K = nn.Linear(EMBEDDING_SIZE, ATTN_DIM_SIZE)
        self.V = nn.Linear(EMBEDDING_SIZE, ATTN_DIM_SIZE)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask = None):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        output = (q @ k.transpose(-2, -1)) / (ATTN_DIM_SIZE ** 0.5)
        if mask is not None:
            output = output.masked_fill(mask == 0, float('-inf'))
        output = self.softmax(output)
        output = output @ v
        return output
        