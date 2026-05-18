"""第五章独立模板：最小 Verify + Benchmark Harness。

目标：
1. 给出一个最小 verify/benchmark 闭环；
2. 强调 warmup / synchronize / 先 verify 再 benchmark 的顺序。
"""

import time
import torch


def verify(kernel_fn, ref_fn, inputs, atol=1e-2, rtol=1e-2):
    out = kernel_fn(*inputs)
    ref = ref_fn(*inputs)
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    max_diff = (out.float() - ref.float()).abs().max().item()
    return ok, max_diff


def benchmark(fn, inputs, warmup=25, iters=100):
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*inputs)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / iters * 1e6


def main():
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    def kernel_fn(t):
        return torch.softmax(t, dim=-1)

    def ref_fn(t):
        return torch.softmax(t, dim=-1)

    ok, max_diff = verify(kernel_fn, ref_fn, (x,))
    latency_us = benchmark(kernel_fn, (x,))

    print({
        "verify_ok": ok,
        "max_diff": max_diff,
        "latency_us": latency_us,
        "dtype": str(x.dtype),
        "device": str(x.device),
    })


if __name__ == "__main__":
    main()
