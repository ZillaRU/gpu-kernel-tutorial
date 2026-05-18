"""第一章独立模板：最小 Triton 向量加法。

目标：
1. 演示 program_id / grid / mask 的最小闭环；
2. 作为后续学习 Triton kernel 地址表达与 launch 方式的起点。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "输入必须在 CUDA/ROCm 设备上"
    assert x.shape == y.shape, "x/y 形状必须一致"
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


def main():
    x = torch.randn(4097, device="cuda", dtype=torch.float32)
    y = torch.randn(4097, device="cuda", dtype=torch.float32)
    out = add(x, y)
    ref = x + y
    print({
        "max_diff": (out - ref).abs().max().item(),
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
    })


if __name__ == "__main__":
    main()
