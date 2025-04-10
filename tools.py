from tests import INPUT_LEN
from torch.nn import functional as F

def pad_input(input):
    return F.pad(input, (0, (INPUT_LEN - len(input))))

def get_mask(input):
    padding_mask = (input == 0)
    encoder_mask_horiz = padding_mask.unsqueeze(0)
    encoder_mask_vert = padding_mask.unsqueeze(1)
    return encoder_mask_horiz | encoder_mask_vert