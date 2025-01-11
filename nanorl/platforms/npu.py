import torch
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from .interface import Platform, PlatformEnum


class NPUPlatform(Platform):
    _enum = PlatformEnum.NPU

    # Device APIs
    @staticmethod
    def device_name(self, device_index=None):
        if device_index is None:
            return "npu"
        return "npu:{}".format(device_index)

    @staticmethod
    def set_device(device_index=None):
        torch.npu.set_device(device_index)

    @staticmethod
    def current_device():
        return torch.npu.current_device()
    
    @staticmethod
    def device_count() -> int:
        return torch.npu.device_count()
    
    @staticmethod
    def current_device_name():
        return "npu:{}".format(torch.npu.current_device())
    
    # RNG APIs
    @staticmethod
    def manual_seed(seed):
        return torch.npu.manual_seed(seed)
    
    # Momory management
    @staticmethod
    def empty_cache():
        torch.npu.empty_cache()
