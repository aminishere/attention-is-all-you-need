import torch
from components.encoder_block import Encoder
from components.decoder_block import Decoder
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model, head,vocab_size):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([Encoder(d_model, head) for _ in range(6)])
        self.decoder_blocks = nn.ModuleList([Decoder(d_model, head) for _ in range(6)])
        self.final_layer=nn.Linear(d_model, vocab_size)
    def forward(self, X, decoder_input, mask=None):
        encoder_output=X
        for layer in self.encoder_blocks:
            encoder_output=layer(encoder_output)
        decoder_output=decoder_input
        for layer in self.decoder_blocks:
            decoder_output=layer(decoder_output, encoder_output, mask)
        logits=self.final_layer(decoder_output)
        return logits
