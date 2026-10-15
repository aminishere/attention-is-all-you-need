import torch
from components.embeddings import Embeddings

model= Embeddings(n=1000, d_model=512)

x= torch.randint(0,1000, (2,5))

y=model(x)


print(y[0].shape)
