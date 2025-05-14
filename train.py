import pickle
import torch

from model import SEQ_LEN, Decoder
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from tools import encode_corpus, get_device, load_pickle_dict, load_txt
from constants import DEFAULT_DEVICE

@dataclass
class TrainingArguments:
    lr: float = field(
        default = 4e-4,
        metadata = {"help": "Learning rate for model training"}
    )
    iter: int = field(
        default = 10,
        metadata = {"help": "Number of iterations for model training"}
    )
    batch_size: int = field(
        default = 1,
        metadata = {"help": "Size of a batch during training"}
    )
    token_to_idx: str = field(
        default = "token_to_idx.pkl",
        metadata = {"help": "Path to token to index pickle vocabulary"}
    )
    corpus: str = field(
        default = "corpus.txt",
        metadata = {"help": "Path to corpus (should be .txt)"}
    )
    save_path: str = field(
        default = "model.pt",
        metadata = {"help": "Path where to save the model (should be either .pt or .pth)"}
    )
    device: str = field(
        default = DEFAULT_DEVICE,
        metadata = {"help": "Device to train the model on. Either 'cpu' or 'cuda' if gpu is available. If not provided will choose automatically"
        "cuda if available"}
    )
    validation: float = field(
        default = 0.1,
        metadata = {"help": "Percentage of data from corpus that will be used as validation dataset"}
    )
    validation_iter: int = field(
        default = 10,
        metadata = {"help": "Number of validation iterations"}
    )
    validation_interval: int = field(
        default = 10,
        metadata = {"help": "Number of training iterations interval between validations"}
    )

def get_idx(input, token_to_idx):
    output = []
    for char in input:
        output.append(token_to_idx[char])
    return output

def get_batch(input, seq_len, batch_size):
    sequences = []
    for batch in range(batch_size):
       sequences.append(torch.tensor(get_idx(input[batch:seq_len + batch], token_to_idx), dtype=torch.long))
    output = torch.stack(sequences)
    return output

def get_random_batch(corpus, seq_len, batch_size):
    rand_idx_in_corpus = torch.randint(high = len(corpus) - seq_len, size = (batch_size,))
    x = torch.stack([corpus[idx: (idx + seq_len)] for idx in rand_idx_in_corpus])
    y = torch.stack([corpus[idx + 1: (idx + seq_len + 1)] for idx in rand_idx_in_corpus])
    return x, y


if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    token_to_idx = load_pickle_dict(args.token_to_idx)
    corpus = load_txt(args.corpus)
    split_idx = int(len(corpus) * (1 - args.validation))
    train_corpus = corpus[: split_idx]
    val_corpus = corpus[split_idx :]
    device = get_device(args.device)
    model = Decoder().to(device)
    print(f"Model loaded to {device}")
    train_corpus = encode_corpus(train_corpus, token_to_idx).to(device)
    val_corpus = encode_corpus(val_corpus, token_to_idx).to(device)
    print("Total number of parameters: ", sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    loss_func = torch.nn.CrossEntropyLoss()
    for iter in range(args.iter):
        x, y = get_random_batch(train_corpus, SEQ_LEN, args.batch_size)
        output = model(x) 
        y = y.view(args.batch_size * SEQ_LEN)
        output = output.view(args.batch_size * SEQ_LEN, - 1)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (iter + 1) % args.validation_interval == 0:
            with torch.no_grad():
                model.eval()
                evals = torch.zeros(args.validation_iter)
                for eval_iter in range(args.validation_iter):
                    x, y = get_random_batch(val_corpus, SEQ_LEN, args.batch_size)
                    eval_output = model(x)
                    y = y.view(args.batch_size * SEQ_LEN)
                    eval_output = eval_output.view(args.batch_size * SEQ_LEN, -1)
                    evals[eval_iter] = loss_func(eval_output, y)
                eval_loss = evals.mean()
                print(f"Iteration {iter} Training Loss: {loss.item()} Evaluation Loss: {eval_loss.item()}")
                model.train()
        else:
            print(f"Iteration {iter} Training Loss: {loss.item()}")


    torch.save(model.state_dict(), args.save_path)