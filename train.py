import pickle
import torch

from model import SEQ_LEN, Decoder
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from tools import load_pickle_dict, load_txt

@dataclass
class TrainingArguments:
    lr: float = field(
        default = 4e-4,
        metadata = {"help": "Learning rate for model training"}
    )
    epoch: int = field(
        default = 1,
        metadata = {"help": "Number of epochs for model training"}
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

def get_idx(input, token_to_idx):
    output = []
    for char in input:
        output.append(token_to_idx[char])
    return output

def get_batch(input, SEQ_LEN, BATCH_SIZE):
    sequences = []
    for batch in range(BATCH_SIZE):
       sequences.append(torch.tensor(get_idx(input[batch:SEQ_LEN + batch], token_to_idx), dtype=torch.long))
    output = torch.stack(sequences)
    return output


if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    token_to_idx = load_pickle_dict(args.token_to_idx)
    corpus = load_txt(args.corpus)

    model = Decoder()
    if (torch.cuda.is_available()):
        device = "cuda"
        model.to("cuda")
        print('Model loaded to cuda')
    else:
        device = "cpu"
        model.to("cpu")
        print('Model loaded to cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(0, args.epoch):
        step = 0
        for left in range(0, len(corpus) - SEQ_LEN, SEQ_LEN):
            substring = corpus[left:(left + SEQ_LEN + args.batch_size - 1)]
            idx = get_idx(substring, token_to_idx)
            batch = get_batch(substring, SEQ_LEN, args.batch_size).to(device)
            output = model(batch) 
            batch = batch.view(args.batch_size * SEQ_LEN)
            output = output.view(args.batch_size * SEQ_LEN, - 1)
            loss = loss_func(output, batch)
            print(f"Epoch {epoch} Step {step} Training Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

    torch.save(model.state_dict(), args.save_path)