from model import EMBEDDING_SIZE, VOCAB_SIZE, AttentionHead, Block, Decoder, FeedForwardNet, MultiHeadAttention
import torch
import torch.nn.functional as F

from tools import encode_corpus
from train import get_random_batch

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

def get_random_batch_test():
    get_random_batch(torch.Tensor([1,2,3,4,5,6,7,8]), 3, 2)

def encode_corpus_test():
    token_to_idx = {'a': 1, 'b': 2, 'c': 3}
    print(encode_corpus("aaabbcc", token_to_idx))

encode_corpus_test()
