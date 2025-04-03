from model import AttentionHead, Block, MultiHeadAttention
import torch

def attention_dims_test():
    sequence_len = 8
    embed_size = 16
    attn_dim_size = 4
    output_dim_size = 4
    x = torch.rand(sequence_len, embed_size)
    attn = AttentionHead(embed_size, attn_dim_size, output_dim_size)
    output = attn(x)
    print(output.shape)

def multihead_dims_test():
    sequence_len = 8
    embed_size = 16
    attn_dim_size = 4
    num_heads = 4
    x = torch.rand(sequence_len, embed_size)
    multihead_attention = MultiHeadAttention(num_heads, attn_dim_size, embed_size)
    print(multihead_attention(x))

def block_test():
    sequence_len = 8
    embed_size = 16
    attn_dim_size = 4
    num_heads = 4
    x = torch.rand(sequence_len, embed_size)
    block = Block(num_heads, attn_dim_size, embed_size)
    print(block(x).shape)

block_test()