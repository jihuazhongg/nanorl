from .interface import Platform, PlatformEnum, UnspecifiedPlatform

current_platform: Platform

is_cuda = False
try:
    import torch
    
    if torch.cuda.device_count() > 0:
        is_cuda = True
except Exception:
    pass


if_npu = False
try:
    import torch_npu  # noqa: F401

    if torch.npu.device_count() > 0:
        is_npu = True
except Exception:
    pass


if is_cuda:
    from .cuda import CudaPlatform
    current_platform = CudaPlatform()
elif is_npu:
    from .npu import NPUPlatform
    current_platform = NPUPlatform()
else:
    current_platform = UnspecifiedPlatform()

__all__ = ['Platform', 'PlatformEnum', 'current_platform']