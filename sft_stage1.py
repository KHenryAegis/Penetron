#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import TrainingArguments, AutoTokenizer, AddedToken
from unsloth import FastLanguageModel
from trl import SFTTrainer
import swanlab
from dataclasses import dataclass, asdict

@dataclass
class TrainingConfig:
    dataset_path: str = "./training_data/sft_think.jsonl"
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "./model/sft"
    max_seq_length: int = 4096
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    eval_size: float = 0
    seed: int = 42
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    swanlab_project: str = "nl2penetration-unsloth"
    load_in_4bit: bool = False

def set_seed(seed_val: int):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

def main():
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    set_seed(config.seed)

    print("Initializing SwanLab...")
    swanlab.init(
        project=config.swanlab_project,
        experiment_name="NL2Pentest-SFT-Unsloth-Full-Tune",
        config=asdict(config),
    )

    try:
        if not os.path.isfile(config.dataset_path):
            raise FileNotFoundError(f"not found：{config.dataset_path}")

        print(f"Loading dataset from: {config.dataset_path}")
        dataset = load_dataset(
            "json",
            data_files={"train": config.dataset_path},
            split="train"
        )
        print(f"Original dataset size: {len(dataset)}")

        # ====== Load model & tokenizer (Unsloth helper) ======
        print(f"Loading model and tokenizer: {config.model_name_or_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            config.model_name_or_path,
            max_seq_length=config.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=config.load_in_4bit,
        )

        # Add special tokens (safer: add strings; AddedToken optional)
        special_tokens = ["<think>", "</think>"]
        # If tokenizer supports AddedToken and you want single_word control, you can use that.
        try:
            # try adding as AddedToken if supported
            added = [AddedToken(t, single_word=True) for t in special_tokens]
            num_added_toks = tokenizer.add_tokens(added)
        except Exception:
            # fallback to plain strings
            num_added_toks = tokenizer.add_tokens(special_tokens)

        if num_added_toks and num_added_toks > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"Added {num_added_toks} new special tokens: {special_tokens}")


        if getattr(tokenizer, "pad_token", None) is None:
            print("Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        # ====== Format conversation data ======
        print("Formatting prompts...")

        def create_conversation(example):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"You are a seasoned Linux command-line expert.\nYour primary mission is to generate accurate and effective terminal commands based on the provided task. Adhere strictly to the required output format.\n\n## Task: {example.get('input', '')}\n\n### Output Format:\nCommand:"},
                {"role": "assistant", "content": "<think>" + example.get("reasoning", "") + "</think>\nCommand: " + example.get("answer", "")}
            ]
            return {"messages": messages}

        conversation_dataset = dataset.map(
            create_conversation,
            remove_columns=[c for c in dataset.column_names if c in ("input", "reasoning", "answer")],
        )

        def format_conversation(example):
            try:
                # apply_chat_template is custom in some tokenizers — keep tokenize=False to get raw text
                formatted_text = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False
                )
                return {"text": formatted_text}
            except Exception as e:
                try:
                    msgs = example.get("messages", [])
                    parts = []
                    for m in msgs:
                        role = m.get("role", "")
                        content = m.get("content", "")
                        parts.append(f"{role}: {content}")
                    return {"text": "\n".join(parts)}
                except Exception as e2:
                    return {"text": ""}

        formatted_dataset = conversation_dataset.map(
            format_conversation,
            remove_columns=conversation_dataset.column_names
        )

        formatted_dataset = formatted_dataset.filter(lambda x: bool(x.get("text", "").strip()))
        print(f"format dataset: {len(formatted_dataset)}")

        print(f"Filtering out samples longer than {config.max_seq_length} tokens...")
        def not_too_long(example):
            try:
                enc = tokenizer(example["text"])
                input_ids = enc.get("input_ids") or enc.get("input_ids", [])
                return len(input_ids) <= config.max_seq_length
            except Exception as e:
                print(f"Tokenization error for a sample, dropping it: {e}")
                return False

        filtered_dataset = formatted_dataset.filter(not_too_long)
        print(f"Filtered dataset size (<= {config.max_seq_length} tokens): {len(filtered_dataset)}")

        if len(filtered_dataset) > 0:
            print("-" * 50)
            print(filtered_dataset[0]["text"])
            print("-" * 50)
        else:
            return

        # ====== TrainingArguments ======
        print("Configuring TrainingArguments...")
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            optim=config.optim,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_ratio=config.warmup_ratio,
            logging_dir=os.path.join(config.output_dir, "logs"),
            logging_steps=config.logging_steps,
            bf16=True,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=2,
            weight_decay=0.01,
            report_to=["swanlab"]
        )

        # ====== SFTTrainer ======
        print("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=filtered_dataset,
            eval_dataset=None,
            max_seq_length=config.max_seq_length,
            dataset_text_field="text",
            args=training_args
        )

        # ====== Start training ======
        print("Starting training...")
        trainer.train()
        print("Training completed.")

        # ====== Save final model ======
        final_model_dir = os.path.join(config.output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        print(f"Saving final model to {final_model_dir} ...")
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        print("Model and tokenizer saved successfully.")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        try:
            swanlab.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main()
