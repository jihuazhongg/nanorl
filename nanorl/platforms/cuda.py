import torch

from .interface import Platform, PlatformEnum


class CudaPlatform(Platform):
    _enum = PlatformEnum.CUDA

    # Device APIs
    @staticmethod
    def device_name(device_index=None):
        if device_index is None:
            return 'cuda'
        return 'cuda:{}'.format(device_index)

    @staticmethod
    def set_device(device_index=None):
        torch.cuda.set_device(device_index)

    @staticmethod
    def current_device():
        return torch.cuda.current_device()
    
    @staticmethod
    def current_device_name():
        return "cuda:{}".format(torch.cuda.current_device())
    
    @staticmethod
    def device_count():
        return torch.cuda.device_count()
    
    # RNG APIs
    @staticmethod
    def manual_seed(seed):
        return torch.cuda.manual_seed(seed)
    
    # Momory management
    @staticmethod
    def empty_cache():
        return torch.cuda.empty_cache()
