from tcl import jit, Tensor

import torch

import random

@jit("cuda")
def add(a: Tensor(shape=(2, 3), dtype="float32"), b: Tensor(shape=(2, 3), dtype="float32")) -> Tensor:  # type: ignore
    # if random.random() < 0.5:
    #     out = a + b
    # else:
    #     out = a * b
    out = a + b
    out = out + a
    print(out)
    return out

@jit("cpu")
def print_hello():
    print("Hello, world!")
    return 0

print_hello()

a = Tensor(shape=(2, 3), dtype="float32")
b = Tensor(shape=(2, 3), dtype="float32")
a.data = torch.ones(size=(2, 3), dtype=torch.float32, device="cuda")
b.data = torch.ones(size=(2, 3), dtype=torch.float32, device="cuda")
print(add(a, b))