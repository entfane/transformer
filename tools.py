import pickle
import torch

from constants import DEFAULT_DEVICE

def load_pickle_dict(dictionary):
    with open(dictionary, 'rb') as f:
        output = pickle.load(f)
    return output

def save_pickle_dict(dictionary, path):
    with open(path, 'wb') as f:
        pickle.dump(dictionary, f)

def load_txt(path, encoding = "utf-8"):
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()
        output = ' '.join(lines)
    return output

def encode_corpus(corpus, token_to_idx):
    encoded_corpus = torch.tensor([token_to_idx[token] for token in corpus], dtype = torch.long)
    return encoded_corpus

def get_device(device):
    if device == DEFAULT_DEVICE:
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cpu":
        return "cpu"
    elif device == "cuda":
        if torch.cuda.is_available():
            return device
        else:
            raise Exception("Cuda is unavailable")
    else:
        raise Exception("Device is unavailable. Choose either 'cpu' or 'cuda' if available")

