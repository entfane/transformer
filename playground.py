import torch

input_ids = torch.tensor([1, 2, 3, 0, 0])  # Example input with padding
x = torch.rand(1, 5, 5)
print(x)
padding_mask = (input_ids != 0)
print(padding_mask)
output = x.masked_fill(padding_mask == 0, float('-inf'))
print(output)