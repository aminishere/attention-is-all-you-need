import torch
import torch.nn as nn
import math

class PosEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model=d_model
    def forward(self, seq_len):
        pos = torch.arange(seq_len).unsqueeze(1)#why - col vector to unsqueeze
        i = torch.arange(0, self.d_model, 2)#why
        PE= torch.zeros(seq_len,self.d_model)
        denominator=torch.exp(-math.log(10000)*i/self.d_model)#cool
        PE[:, 0::2]=torch.sin(pos*denominator)
        PE[:,1::2]=torch.cos(pos*denominator)

        return PE
