import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DFF = 1536
SEQ_LEN = 256
EMBEDDING_SIZE = 384
VOCAB_SIZE = 65
NUM_BLOCKS = 6
NUM_HEADS = 6
ATTN_DIM_SIZE = EMBEDDING_SIZE // NUM_HEADS
N = 10000
DROPOUT = 0.2

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.register_buffer('pos_enc', self.__generate_pos_encoding())
        self.blocks = nn.ModuleList([Block() for i in range(NUM_BLOCKS)])
        self.l = nn.Linear(EMBEDDING_SIZE, VOCAB_SIZE)

    def forward(self, x):
        input = self.embedding(x)
        model_device = next(self.parameters()).device
        input += self.pos_enc.to(model_device)
        for block in self.blocks:
            input = block(input)
        input = self.l(input)
        return input
    
    def generate(self, x, max_new_tokens):
        for i in range(max_new_tokens):
            x = x[:, -SEQ_LEN : ]
            logits = self.forward(x)
            last_token_logits = logits[:, -1, :]
            outputs = F.softmax(last_token_logits, dim = -1)
            output = torch.multinomial(outputs, num_samples = 1)
            x = torch.cat((x, output), dim=-1)
        return x
    
    def __generate_pos_encoding(self):
        pos_encoding = torch.zeros((SEQ_LEN, EMBEDDING_SIZE), dtype=torch.float, requires_grad=False)
        for i in range(SEQ_LEN):
            for k in range(EMBEDDING_SIZE // 2):
                value = i / pow(N, (2 * k / EMBEDDING_SIZE))
                pos_encoding[i, k * 2] = math.sin(value)
                pos_encoding[i, k * 2 + 1] = math.cos(value)
        return pos_encoding
    

class Block(nn.Module): 

    def __init__(self):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(EMBEDDING_SIZE)
        self.ff = FeedForwardNet()
        self.norm2 = nn.LayerNorm(EMBEDDING_SIZE)
        self.register_buffer('tril', torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)))
    
    def forward(self, x):

        output = self.norm1(x)
        output = self.masked_multi_head_attention(output, mask = self.tril)
        output = x + output

        x = output
        output = self.norm2(output)
        output = self.ff(output)
        output = x + output
        
        return output

class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(EMBEDDING_SIZE, DFF)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(DFF, EMBEDDING_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead() for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x, mask = None):
        output = torch.cat([head(x, mask) for head in self.heads], dim = -1)
        output = self.proj(output)
        output = self.dropout(output)
        return output
        
        

class AttentionHead(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(EMBEDDING_SIZE, ATTN_DIM_SIZE)
        self.K = nn.Linear(EMBEDDING_SIZE, ATTN_DIM_SIZE)
        self.V = nn.Linear(EMBEDDING_SIZE, ATTN_DIM_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x, mask = None):
        B, T, C = x.shape
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        output = (q @ k.transpose(-2, -1)) * ATTN_DIM_SIZE ** -0.5
        if mask is not None:
            output = output.masked_fill(mask[:T, :T] == 0, float('-inf'))
        output = F.softmax(output, dim = -1)
        output = self.dropout(output)
        output = output @ v
        return output
        