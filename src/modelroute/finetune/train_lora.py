"""LoRA / QLoRA supervised fine-tuning of a causal LM for ticket classification.

Requires a CUDA GPU and ``pip install 'modelroute[finetune]'``. Not executed in this repo's CI
(no GPU / no model downloads) - see README "What was run here".

Usage::

    modelroute export-sft                       # writes data/generated/sft_{train,val}.jsonl
    python -m modelroute.finetune.train_lora configs/lora_qwen2.5-0.5b.yaml

Key choices
-----------
* Chat-template formatting with **loss only on the assistant JSON** (prompt tokens masked with
  -100), so capacity is spent on the label, not on copying the ticket.
* QLoRA: 4-bit NF4 base weights + double quantisation, bf16 compute, LoRA on all linear
  projections (attention + MLP).
* Eval hook: besides eval loss, a callback greedily generates labels for a fixed val subset and
  logs exact-match intent accuracy / JSON-validity - loss alone is a poor proxy for accuracy.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from .formatting import FinetuneConfig

log = logging.getLogger("modelroute.finetune")


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def tokenize_example(tokenizer: Any, messages: list[dict[str, str]],
                     max_len: int) -> dict[str, Any]:
    """Tokenise a chat example; mask everything except the assistant answer."""
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
                                               add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    full_ids = full_ids[:max_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def main(config_path: str) -> None:  # pragma: no cover - requires GPU + model weights
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    cfg = FinetuneConfig.from_yaml(Path(config_path))
    t = cfg.train
    torch.manual_seed(t.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if cfg.method == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=quant, device_map="auto",
        torch_dtype=torch.bfloat16 if t.bf16 else torch.float16)
    if cfg.method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=t.gradient_checkpointing)
    elif t.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    train_rows = _load_jsonl(cfg.train_file)
    val_rows = _load_jsonl(cfg.val_file)
    train_ds = Dataset.from_list(train_rows).map(
        lambda ex: tokenize_example(tokenizer, ex["messages"], t.max_seq_len),
        remove_columns=["id", "messages"])
    val_ds = Dataset.from_list(val_rows).map(
        lambda ex: tokenize_example(tokenizer, ex["messages"], t.max_seq_len),
        remove_columns=["id", "messages"])
    gen_subset = val_rows[: t.eval_generate_samples]

    class GenerationEvalCallback(TrainerCallback):
        """Greedy-decode a fixed val subset and log intent exact-match + JSON validity."""

        def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            model.eval()
            correct = valid = 0
            for row in gen_subset:
                prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False,
                                                       add_generation_prompt=True)
                enc = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)
                text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
                gold = json.loads(row["messages"][-1]["content"])
                try:
                    pred = json.loads(text.strip())
                    valid += 1
                    correct += int(pred.get("intent") == gold["intent"])
                except json.JSONDecodeError:
                    pass
            n = max(1, len(gen_subset))
            metrics = {"eval_intent_acc": correct / n, "eval_json_valid": valid / n}
            log.info("generation eval %s", metrics)
            state.log_history.append({**metrics, "step": state.global_step})
            model.train()

    args = TrainingArguments(
        output_dir=cfg.output_dir, num_train_epochs=t.epochs, learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_batch_size,
        per_device_eval_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum_steps, warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay, lr_scheduler_type=t.lr_scheduler,
        eval_strategy="steps", eval_steps=t.eval_steps, save_strategy="steps",
        save_steps=t.eval_steps, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", logging_steps=20, bf16=t.bf16, fp16=not t.bf16,
        optim="paged_adamw_8bit" if cfg.method == "qlora" else "adamw_torch",
        report_to="none", seed=t.seed)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                                           label_pad_token_id=-100),
                      callbacks=[GenerationEvalCallback()])
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    log.info("saved LoRA adapter to %s", cfg.output_dir)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/lora_qwen2.5-0.5b.yaml")
