"""第二章独立模板：最小 Fused RMSNorm。

目标：
1. 演示 fusion、fp32 reduce、rsqrt 的基本写法；
2. 作为后续继续调 BLOCK_SIZE / 向量化 / multi-row 的起点。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, hidden_dim, stride_row, eps,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_dim

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / hidden_dim
    inv_rms = tl.rsqrt(mean_sq + eps)
    y = x * inv_rms * w

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.float16), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "输入和权重必须在 GPU 上"
    assert x.ndim == 2, "模板默认输入为 [rows, hidden_dim]"
    rows, hidden_dim = x.shape
    out = torch.empty_like(x)
    block_size = triton.next_power_of_2(hidden_dim)
    rmsnorm_kernel[(rows,)](x, weight, out, hidden_dim, x.stride(0), eps, BLOCK_SIZE=block_size)
    return out


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rms * weight.float()).to(x.dtype)


def main():
    x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
    w = torch.randn(1024, device="cuda", dtype=torch.float16)
    out = rmsnorm(x, w)
    ref = rmsnorm_ref(x, w)
    print({
        "max_diff": (out - ref).abs().max().item(),
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
    })


if __name__ == "__main__":
    main()
