from typing import Any, Dict, Optional

from ..params import get_train_args
from .pt import run_pt


def run_tuner(args: Optional[Dict[str, Any]] = None) -> None:
    # callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    if finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args)
    elif finetuning_args.stage == "sft":
        # run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
        pass
    else:
        raise ValueError(f"Unknow task: {finetuning_args.stage}")
