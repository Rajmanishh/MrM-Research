from models.perceiver_io import PerceiverIO
import torch

model = PerceiverIO(C=128, N=64, D=128)

x = torch.randn(2, 50, 128)

out = model(x)

print(out.shape)