# from typing import Optional

# from transformers import Trainer, ProcessorMixin

# from ...params import FinetuningArguments


# class CustomTrianer(Trainer):
    
#     def __init__(
#         self,
#         finetuning_args: "FinetuningArguments",
#         processor: Optional["ProcessorMixin"], 
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.finetuning_args = finetuning_args

#         if processor is not None:
#             self.add_callback(SaveProcessorCallback(processor))

    