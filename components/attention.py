import torch
import math

def scaled_dot_product_attention(Q,K,V):
    d_k= Q.size(-1)
    prod= Q@K.transpose(-2,-1)
    prod=prod/math.sqrt(d_k)
    softmax_prod=torch.softmax(prod, dim=-1)
    prod_2=softmax_prod@V
    return prod_2

class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model):
        super().__init__()
        self.head=head
        self.d_model=d_model
        self.d_k=d_model/head
        self.Wq=nn.Linear(self.d_model, self.d_model)
        self.Wk=nn.Linear(self.d_model, self.d_model)
        self.Wv=nn.Linear(self.d_model, self.d_model)
        self.Wo=nn.Linear(self.d_model, self.d_model)

     def forward(self, X):
         Q=self.Wq(X)
         Q=Q.reshape(Q.size(0), Q.size(1), self.head, self.d_k)
         K=self.Wk(X)
         K=K.reshape(K.size(0), K.size(1), self.head, self.d_k)
         V=self.Wv(X)
         V=V.reshape(V.size(0), V.size(1), self.head, self.d_k)
         Q = Q.transpose(1, 2) #8*seq_len
         K = K.transpose(1, 2)
         V = V.transpose(1, 2)
         attention =scaled_dot_product_attention(Q,K,V)
         attention = attention.transpose(1, 2)
         concat = attention.reshape(attention.size(0), attention.size(1), self.d_model)
         output=self.Wo(concat)
         return output


