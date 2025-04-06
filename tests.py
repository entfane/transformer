from model import AttentionHead, Block, Encoder, MultiHeadAttention
import torch

def attention_dims_test():
    sequence_len = 8
    embed_size = 16
    attn_dim_size = 4
    output_dim_size = 4
    x = torch.rand(4, sequence_len, embed_size)
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

def encoder_test():
    vocab_size = 16
    embed_size = 32
    num_blocks = 4
    num_heads = 4
    attention_dim_size = 4
    seq_len = 8
    encoder = Encoder(vocab_size,embed_size, num_blocks, num_heads, attention_dim_size, seq_len)
    x = torch.randint(low = 0, high = vocab_size - 1, size = (1, seq_len))
    output = encoder(x)
    print(output.shape)
# attention_dims_test()
encoder_test()