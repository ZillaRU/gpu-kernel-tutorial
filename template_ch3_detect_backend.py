"""第三章独立模板：最小架构/后端检测脚本。

目标：
1. 明确区分 CUDA 与 ROCm 的运行时语义；
2. 提醒读者不要把 torch.cuda 接口直接等同于 NVIDIA 平台。
"""

import torch


def detect_backend() -> dict:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "backend": "none",
            "device_name": None,
            "device_count": 0,
        }

    backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    return {
        "available": True,
        "backend": backend,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": getattr(torch.version, "hip", None),
    }


def main():
    print(detect_backend())


if __name__ == "__main__":
    main()
