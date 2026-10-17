import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward

class Decoder(nn.Module):
    def __init__(self, d_model,head):
        super().__init__()
        self.d_model = d_model
        self.masked_attention=MultiHeadAttention(head, d_model)
        self.cross_attention=MultiHeadAttention(head, d_model)
        self.ln1= nn.LayerNorm(d_model)
        self.ffn=FeedForward(d_model)
        self.ln2= nn.LayerNorm(d_model)
        self.ln3= nn.LayerNorm(d_model)



    def forward(self, X, encoder_output, mask=None):
        l1=self.masked_attention(X, mask=mask)#masked
        l2=self.ln1(l1+X)
        l3= self.cross_attention(l2, encoder_output, encoder_output)#input from encoder-K,V
        l4=self.ln2(l3+l2)
        l5=self.ffn(l4)
        l6=self.ln3(l5+l4)
        return l6

