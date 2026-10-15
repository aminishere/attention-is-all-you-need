import torch
from components.pos_encoding import PosEncoding

model = PosEncoding(d_model=512)


output = model(seq_len=5)

print(output.shape)
print(output[0, :10])
print(output[1, :10])
