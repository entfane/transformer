from model import AttentionHead
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

attention_dims_test()