from model import EMBEDDING_SIZE, VOCAB_SIZE, AttentionHead, Block, Decoder, FeedForwardNet, MultiHeadAttention
import torch
import torch.nn.functional as F

SEQ_LEN = 8
BATCH_SIZE = 2

def attention_test():
    input = torch.randint(low = 0, high = VOCAB_SIZE, size = (BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE))
    input = input.to(dtype=torch.float)
    tril = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
    print(tril)
    ah = AttentionHead()
    output = ah(input, tril)
    print(output)


def multi_head_attention_test():
    input = torch.randint(low = 0, high = VOCAB_SIZE, size = (BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE))
    input = input.to(dtype=torch.float)
    tril = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
    mh = MultiHeadAttention()
    output = mh(input, tril)
    print(output.shape)

def ffwd_test():
    input = torch.randint(low = 0, high = VOCAB_SIZE, size = (BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE))
    input = input.to(dtype=torch.float)
    ffwd = FeedForwardNet()
    output = ffwd(input)
    print(output)


def block_test():
    input = torch.randint(low = 0, high = VOCAB_SIZE, size = (BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE))
    input = input.to(dtype=torch.float)
    block = Block()
    output = block(input)
    print(output.shape)

def decoder_test():
    input = torch.randint(low = 0, high = VOCAB_SIZE, size = (BATCH_SIZE, SEQ_LEN))
    print(input)
    decoder = Decoder()
    output = decoder(input)
    print(output[0][-1])

decoder_test()
