from enum import Enum
from typing import Dict, List, Sequence, Set, Union

from loguru import logger
from datasets import DatasetDict, Dataset, IterableDataset, concatenate_datasets

from ..params import DataArguments


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Merges multiple datasets to a unified dataset.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)
    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", seed: int
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    if data_args.streaming:
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
        val_set = dataset.take(int(data_args.val_size))
        train_set = dataset.skip(int(data_args.val_size))
        return DatasetDict({"train": train_set, "validation": val_set})
    else:
        val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        dataset = dataset.train_test_split(test_size=val_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
