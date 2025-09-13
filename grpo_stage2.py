import os
import json
import re
import shlex
import traceback
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import swanlab
import torch
from datasets import Dataset
from transformers import AddedToken
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

max_seq_length = 4096

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
COMMAND_PATTERN = re.compile(r"</think>.*?Command:\s*(.*?)(?:\n|$)", re.DOTALL | re.IGNORECASE)


def normalize_command(cmd):
    try:
        cmd = re.sub(r'\s+', ' ', cmd).strip()
        cmd = cmd.replace('"', "'")
        cmd = re.sub(r'(\s*[|&;]\s*)+$', '', cmd)

        try:
            parts = shlex.split(cmd)
        except Exception:
            parts = cmd.split()

        if not parts:
            return ""

        command_name = parts[0]
        args = parts[1:]

        options = []
        positional = []
        option_values = defaultdict(list)
        i = 0
        while i < len(args):
            if args[i].startswith('-'):
                if '=' in args[i]:
                    opt, val = args[i].split('=', 1)
                    options.append(opt)
                    option_values[opt].append(val)
                    i += 1
                elif i + 1 < len(args) and not args[i + 1].startswith('-'):
                    options.append(args[i])
                    option_values[args[i]].append(args[i + 1])
                    i += 2
                else:
                    options.append(args[i])
                    i += 1
            else:
                positional.append(args[i])
                i += 1

        unique_options = []
        seen = set()
        for opt in options:
            if opt not in seen:
                unique_options.append(opt)
                seen.add(opt)

        normalized = [command_name]
        for opt in sorted(unique_options):
            normalized.append(opt)
            for val in option_values[opt]:
                normalized.append(val)

        normalized.extend(positional)
        return ' '.join(normalized)
    except Exception as e:
        return cmd


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        try:
            response = completion[0]["content"]
        except Exception:
            rewards.append(0.0)
            continue

        score = 0.0
        think_match = THINK_PATTERN.search(response)
        if think_match and think_match.group(1).strip():
            score += 0.25

        if COMMAND_PATTERN.search(response):
            score += 0.25

        rewards.append(score)
    return rewards


def accuracy_reward(prompts, completions, answer, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        try:
            response = completion[0]["content"]
        except Exception:
            rewards.append(0.0)
            continue

        true_answer = answer[i] if isinstance(answer, (list, tuple)) and i < len(answer) else answer
        score = 0.0

        gen_command_match = COMMAND_PATTERN.search(response)
        if not gen_command_match:
            rewards.append(score)
            continue

        gen_command = gen_command_match.group(1).strip()
        gen_command = normalize_command(gen_command)
        true_command = normalize_command(true_answer)

        if not true_command:
            rewards.append(score)
            continue

        if gen_command == true_command:
            score += 1.0
        elif gen_command:
            gen_words = set(gen_command.split())
            true_words = set(true_command.split())
            intersection = gen_words & true_words
            union_size = len(gen_words) + len(true_words) - len(intersection)
            if union_size > 0:
                similarity = len(intersection) / union_size
                if similarity >= 0.5:
                    score += similarity * 0.5

        rewards.append(score)
    return rewards


def load_dataset_from_jsonl(dataset_path):
    SYSTEM_PROMPT = "You are a helpful assistant."
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            user_prompt = (
                "You are a seasoned Linux command-line expert. Your primary mission is to generate accurate and effective terminal commands based on the provided task.\n\n"
                f"## Task:{example.get('input','')}\n\n"
                "### Output Format:\nCommand:"
            )
            message = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "answer": example.get("answer", "")
            }
            dataset.append(message)
    return Dataset.from_list(dataset)


def create_training_config(max_prompt_length, max_completion_length, output_dir):
    training_args = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        save_steps=50,
        save_total_limit=3,
        num_train_epochs=2,
        report_to="swanlab",
        output_dir=output_dir,
    )
    return training_args


def main():
    try:
        dataset_path = "./training_data/grpo_train.jsonl"
        output_dir = "./model/grpo"
        os.makedirs(output_dir, exist_ok=True)

        dataset = load_dataset_from_jsonl(dataset_path)
        print(f"Loaded {len(dataset)} examples from {dataset_path}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/home/kh/nl2penetration/sft/final_model",
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            fast_inference=True,
            gpu_memory_utilization=0.8,
            device_map="auto"
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=64,
            use_gradient_checkpointing=True,
            random_state=3407,
        )

        special_tokens = ["<think>", "</think>"]
        try:
            added = [AddedToken(t, single_word=True) for t in special_tokens]
            num_added = tokenizer.add_tokens(added)
        except Exception:
            num_added = tokenizer.add_tokens(special_tokens)
        if num_added and num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
            print(f"Added {num_added} tokens: {special_tokens}")

        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        def compute_tokens_and_length(example):
            try:
                toks = tokenizer.apply_chat_template(example["prompt"], add_generation_prompt=True, tokenize=True)
                if isinstance(toks, dict) and "input_ids" in toks:
                    toks = toks["input_ids"]
                L = len(toks) if toks is not None else 0
                return {"tokens": toks, "L": L}
            except Exception as e:
                print(f"[compute_tokens_and_length] tokenization failed for one sample: {e}")
                return {"tokens": [], "L": 0}

        print("Tokenizing prompts to get lengths (this may take a while)...")
        tokenized = dataset.map(compute_tokens_and_length, batched=False)

        Ls = tokenized["L"]
        if len(Ls) == 0:
            raise RuntimeError("No samples in dataset after tokenization.")
        maximum_length = int(np.max(Ls))
        print("Max prompt token length in dataset = ", maximum_length)

        keep_indices = [i for i, L in enumerate(Ls) if L <= max_seq_length]
        print(f"Keeping {len(keep_indices)}/{len(Ls)} samples with prompt length <= {max_seq_length} tokens.")
        if len(keep_indices) == 0:
            raise RuntimeError(f"No samples <= {max_seq_length} tokens.")

        dataset = dataset.select(keep_indices)

        kept_Ls = [Ls[i] for i in keep_indices]
        max_prompt_length = int(np.max(kept_Ls)) + 1  
        max_completion_length = max_seq_length - max_prompt_length
        if max_completion_length <= 0:
            raise RuntimeError("Computed max_completion_length <= 0.")
        print(f"max_prompt_length = {max_prompt_length}, max_completion_length = {max_completion_length}")

        training_args = create_training_config(max_prompt_length, max_completion_length, output_dir)

        try:
            swan_config = asdict(training_args)
        except Exception:
            try:
                swan_config = dict(training_args.__dict__)
            except Exception:
                swan_config = {}
        swanlab.init(
            project="nl2pentest_grpo",
            experiment_name="NL2Pentest-GRPO",
            config=swan_config,
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[format_reward, accuracy_reward],
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        final_model_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        print(f"Saving final model to {final_model_dir} ...")
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        print("Model and tokenizer saved successfully.")

    except Exception as e:
        traceback.print_exc()
    finally:
        try:
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
