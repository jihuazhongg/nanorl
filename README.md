# NanoRL

Nano version of openrlhf, supports running on ascend A2 800T, for study only.

---

For ppo_trainer.py:

- no sample packing
- no ema model
- no lora & quant
- no remote reward model
- no tensorboard
- only support gae advantage_estimator

----

for dpo_trainer.py
- no fa
- no lora & nf4 quant
- no packing sample

