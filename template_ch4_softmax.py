"""第四章独立模板：最小 Softmax Kernel。

目标：
1. 演示 safe softmax 的基本写法；
2. 演示 memory-bound 小算子的最小 Triton 实现骨架。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, x_ptr, n_cols, stride_row, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.float16), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2, "模板默认输入为 GPU 上的二维张量"
    rows, n_cols = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(n_cols)
    softmax_kernel[(rows,)](out, x, n_cols, x.stride(0), BLOCK_SIZE=block_size)
    return out


def main():
    x = torch.randn(256, 1024, device="cuda", dtype=torch.float16)
    out = softmax(x)
    ref = torch.softmax(x.float(), dim=-1).to(x.dtype)
    print({
        "max_diff": (out - ref).abs().max().item(),
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
    })


if __name__ == "__main__":
    main()
