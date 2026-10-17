import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model= d_model
        self.ff1=nn.Linear(d_model, 2048)
        self.ff2=nn.Linear(2048, d_model)
    def forward(self, X):
         output1=torch.relu(self.ff1(X))
         output2=self.ff2(output1)
         return output2
