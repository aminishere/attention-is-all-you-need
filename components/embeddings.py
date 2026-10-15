import torch
import torch.nn as nn
import math



class Embeddings(nn.Module):
    def __init__(self, n, d_model):
        super().__init__()
        self.d_model=d_model
        self.embedding = nn.Embedding(n, d_model)
    def forward(self, input):
        input_embedding = self.embedding(input)
        scaled_embedding=input_embedding*math.sqrt(self.d_model)
        return scaled_embedding
