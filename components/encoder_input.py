from embeddings import Embedding
from pos_encoding import PosEncoding

class EncoderInput(nn.Module):
    def __init__(self, d_model,n):
        super().__init__()
        self.d_model=d_model
        self.embedding= Embedding(n, d_model)
        self.pos_encoding=PosEncoding(d_model)

    def forward(self, X):
        l1=self.embedding(X)
        l2=self.pos_encoding(X)
        l3=l1+l2
        return l3
