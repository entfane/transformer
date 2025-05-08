import pickle
import torch

from model import SEQ_LEN, Decoder

LR = 4e-4
EPOCHS = 1
BATCH_SIZE = 2
MAX_STEPS = None

def get_idx(input, token_to_idx):
    output = []
    for char in input:
        output.append(token_to_idx[char])
    return output

with open('token_to_idx.pkl', 'rb') as f:
    token_to_idx = pickle.load(f)

with open('idx_to_token.pkl', 'rb') as f:
    idx_to_token = pickle.load(f)

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    single_line = ' '.join(lines)

model = Decoder()



if (torch.cuda.is_available()):
    device = "cuda"
    model.to("cuda")
    print('Model loaded to cuda')
else:
    device = "cpu"
    model.to("cpu")
    print('Model loaded to cpu')

optimizer = torch.optim.Adam(model.parameters(), lr = LR)
loss_func = torch.nn.CrossEntropyLoss()

def get_batch(input, SEQ_LEN, BATCH_SIZE):
    sequences = []
    for batch in range(BATCH_SIZE):
       sequences.append(torch.tensor(get_idx(input[batch:SEQ_LEN + batch], token_to_idx), dtype=torch.long))
    output = torch.stack(sequences)
    return output


for epoch in range(0, EPOCHS):
    step = 0
    for left in range(0, len(single_line) - SEQ_LEN, SEQ_LEN):
        if (MAX_STEPS is not None) and (step >= MAX_STEPS):
            break
        substring = single_line[left:(left + SEQ_LEN + BATCH_SIZE - 1)]
        idx = get_idx(substring, token_to_idx)
        batch = get_batch(substring, SEQ_LEN, BATCH_SIZE).to(device)
        output = model(batch) # B, T, V
        # B, T
        batch = batch.view(BATCH_SIZE * SEQ_LEN)
        output = output.view(BATCH_SIZE * SEQ_LEN, - 1)
        loss = loss_func(output, batch)
        print(f"Loss at step {step}: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

torch.save(model.state_dict(), "transformer.pt")


    
        