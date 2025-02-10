import torch 
import torch.nn as nn 
from torch.nn import functional as F 

loss = nn.CrossEntropyLoss()
device = torch.device("cpu")
a = torch.rand(64, 4, 128, 128).to(device)
b = torch.randint(0, 5, size = (64,128, 128)).to(device)
print(a.shape, b.shape)
print(loss(a, b))

