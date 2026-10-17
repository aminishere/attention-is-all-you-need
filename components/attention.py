import torch
import math

def scaled_dot_product_attention(Q,K,V):
    d_k= Q.size(-1)
    prod= Q@K.transpose(-2,-1)
    prod=prod/math.sqrt(d_k)
    softmax_prod=torch.softmax(prod, dim=-1)
    prod_2=softmax_prod@V
    return prod_2
