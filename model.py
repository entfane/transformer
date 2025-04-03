import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        pass
        
    def encode(self, x):
        pass

    def decode(self, x):
        pass

class Decoder(nn.Module):

    def __init__():
        super().__init__()

    def forward(self, x):
        pass

class Encoder(nn.Module):
    
    def __init__():
        super().__init__()

    def forward(self, x):
        pass