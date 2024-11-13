from typing import Callable, Literal, Tuple
from functools import partial

from transformers import PreTrainedTokenizer

from ..params import DataArguments
from .template import Template
from .processors.pretrain import preprocess_pretrain_dataset
from .processors.unsupervised import (
    print_unsupervised_dataset_example,
    preprocess_unsupervised_dataset,
)
from .processors.supervised import (
    preprocess_packed_supervised_dataset, 
    preprocess_supervised_dataset,
    print_supervised_dataset_example,
)



def get_preprocess_and_print_func(
    data_args: "DataArguments",
    stage: Literal["pt", "sft"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    do_generate: bool = False,
) -> Tuple[Callable, Callable]:
    if stage == "pt":
        preprocess_func = partial(
            preprocess_pretrain_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
    elif stage == "sft" and not do_generate:
        # pack的方式待解读
        if data_args.packing:
            preprocess_func = partial(
                preprocess_packed_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                data_args=data_args,
            )
        else:
            preprocess_func = partial(
                preprocess_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                data_args=data_args,
            )

        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    else:
        preprocess_func = partial(
            preprocess_unsupervised_dataset,
            template=template,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function
