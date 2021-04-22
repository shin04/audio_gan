from torchsummary import summary
import torch

from models.Conformer import ConformerBlock
from models.Discriminator import Discriminator
from models.Generator import Generator

# model = ConformerBlock(dim=64)
# summary(model, (16, 64))  # summary(model, input_shape)
# x = torch.randn(1, 16, 64)
# print(x.shape)
# print(model(x))

# d = Discriminator()
# summary(d, (1, 128, 216))

g = Generator(height=24, width=24)
summary(g, (100, ))
