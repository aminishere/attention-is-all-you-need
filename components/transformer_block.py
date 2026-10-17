import torch
from encoder_block import Encoder
from decoder_block import Decoder

class TransformerBlock(nn.Module):
    def __init__(self, d_model, head):
        super().__init__()
        self.encoder_block = Encoder(d_model, head)
        self.decoder_block= Decoder(d_model, head)

     def forward(self, X):

