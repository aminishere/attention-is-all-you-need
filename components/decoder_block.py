import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward

class Decoder(nn.Module):
    def __init__(self, d_model,head):
        super().__init__()
        self.d_model = d_model
        self.multi_head_attention=MultiHeadAttention(head, d_model)
        self.ln1= nn.LayerNorm(d_model)
        self.ffn=FeedForward(d_model)
        self.ln2= nn.LayerNorm(d_model)
        self.ln2= nn.LayerNorm(d_model)



    def forward(self, X):
        l1=self.multi_head_attention(X)#masked
        l2=self.ln1(l1+X)

