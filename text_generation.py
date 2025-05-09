import torch
import pickle

from model import SEQ_LEN, Decoder

PATH = "transformer.pt"

with open('idx_to_token.pkl', 'rb') as f:
    idx_to_token = pickle.load(f)

def load_model():
    model = Decoder()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model


def decode(output):
    B, T = output.shape
    outputs = []
    for btch in range(B):
        batch_tokens = ""
        for time in range(T):
            character = idx_to_token[output[btch, time].item()]
            batch_tokens += character
        outputs.append(batch_tokens)
    return outputs

model = load_model()
model.to("cuda")
input = torch.ones((1, SEQ_LEN), dtype=torch.int).to("cuda")
outputs = model.generate(input, 128)
print(decode(outputs))