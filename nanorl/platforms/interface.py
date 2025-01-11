import enum
from typing import Tuple

import torch


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    NPU = enum.auto()
    UNSPECIFIED = enum.auto()


class Platform:
    _enum: PlatformEnum

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.NPU
    
    # Device APIs
    @staticmethod
    def device_name(device_index=None):
        raise NotImplementedError

    @staticmethod
    def set_device(device_index=None):
        raise NotImplementedError

    @staticmethod
    def current_device():
        raise NotImplementedError
    
    @staticmethod
    def current_device_name():
        raise NotImplementedError
    
    @staticmethod
    def device_count():
        raise NotImplementedError
    
    # RNG APIs
    def manual_seed(seed):
        raise NotImplementedError
    
    # Momory management
    @staticmethod
    def empty_cache():
        raise NotImplementedError


    # @staticmethod
    # def get_device_capability(device_id: int = 0) -> Tuple[int, int]:
    #     raise NotImplementedError

    # @staticmethod
    # def get_device_name(device_id: int = 0) -> str:
    #     raise NotImplementedError

    # @staticmethod
    # def inference_mode():
    #     """A device-specific wrapper of `torch.inference_mode`.

    #     This wrapper is recommended because some hardware backends such as TPU
    #     do not support `torch.inference_mode`. In such a case, they will fall
    #     back to `torch.no_grad` by overriding this method.
    #     """
    #     return torch.inference_mode(mode=True)


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
