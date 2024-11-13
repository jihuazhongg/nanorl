from transformers import (
    Trainer,
    DataCollatorForLanguageModeling, 
    Seq2SeqTrainingArguments,
)

from ...params import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
)
from ...model import load_model, load_tokenizer
from ...data import (
    get_template_and_fix_tokenizer,
    get_dataset,
)
# from .trainer import CustomTrianer

def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
):
    tokenizer = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args) # fix tokenizer 需要再这里做嘛？
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", tokenizer=tokenizer)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    # Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    # are not all of the same length.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        **dataset_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
