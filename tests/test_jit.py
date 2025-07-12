from tcl import jit, Tensor

import torch


@jit("cuda")
def two_add_cuda(a: Tensor(shape=(2, 3), dtype="float32"), b: Tensor(shape=(2, 3), dtype="float32")) -> Tensor:  # type: ignore
    out = a + b
    out = out + a
    return out


@jit("cpu")
def two_add_cpu(a: Tensor(shape=(2, 3), dtype="float32"), b: Tensor(shape=(2, 3), dtype="float32")) -> Tensor:  # type: ignore
    out = a + b
    out = out + a
    return out


@jit("cpu")
def print_hello():
    print("Hello, world!")
    return 0


def test_two_add_cuda():
    a = Tensor(shape=(2, 3), dtype="float32")
    b = Tensor(shape=(2, 3), dtype="float32")
    a_torch = torch.ones(size=(2, 3), dtype=torch.float32, device="cuda")
    b_torch = torch.ones(size=(2, 3), dtype=torch.float32, device="cuda")
    a.data = a_torch
    b.data = b_torch
    correct_result = a_torch + b_torch + a_torch
    result = two_add_cuda(a, b)
    assert torch.allclose(
        torch.from_numpy(result.numpy()).to(torch.device("cuda:0")),
        correct_result,
        atol=1e-5,
    ), "CUDA two_add failed"


def test_two_add_cpu():
    a = Tensor(shape=(2, 3), dtype="float32")
    b = Tensor(shape=(2, 3), dtype="float32")
    a_torch = torch.ones(size=(2, 3), dtype=torch.float32, device="cpu")
    b_torch = torch.ones(size=(2, 3), dtype=torch.float32, device="cpu")
    a.data = a_torch
    b.data = b_torch
    correct_result = a_torch + b_torch + a_torch
    result = two_add_cpu(a, b)
    assert torch.allclose(
        torch.from_numpy(result.numpy()).to(torch.device("cpu")),
        correct_result,
        atol=1e-5,
    ), "CPU two_add failed"


def test_print_hello():
    result = print_hello()
    assert result.numpy() == 0, "print_hello did not return 0"