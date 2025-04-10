from model import AttentionHead, Block, Encoder, MultiHeadAttention
import torch
import torch.nn.functional as F

INPUT_LEN = 128

def attention_dims_test():
    sequence_len = 8
    embed_size = 16
    attn_dim_size = 4
    output_dim_size = 4
    x = torch.rand(1, sequence_len, embed_size)
    attn = AttentionHead(embed_size, attn_dim_size, output_dim_size)
    print(x)
    mask = torch.zeros(sequence_len)
    mask[0] = 1
    mask[1] = 1
    mask[3] = 1
    # print(mask)
    output = attn(x, mask)
    # print(x)
    # print(output)
    # print(output.shape)

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
    x = torch.randint(low = 0, high = vocab_size - 1, size = (16, seq_len))
    output = encoder(x)
    print(output.shape)


def test_tokenizer():
    input_size = 16
    input = torch.ones(12)
    I = input.shape[0]
    input = input.view(1, I)
    print(input)
    input_padded = F.pad(input, (0, (16 - I), 0, 0))
    mask = torch.ones(I)
    mask = torch.cat((mask, torch.zeros(16 - I)))
    print(mask)
    print(input_padded)
    print(input_padded.masked_fill(mask == 0, float('-inf')))


attention_dims_test()
# x = torch.Tensor([[[1, 2, 3, 4], [2, 3, 4, 5], [4, 4, 4, 4]]])
# mask = torch.Tensor([[1, 1, 0]])
# print(x.masked_fill(mask == 0, float('-inf')))
