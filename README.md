# Penetron

[English](#english) | [中文](#chinese)

<a name="english"></a>

## 🇬🇧 English Description

**Penetron** is a framework designed to train and evaluate Large Language Models (LLMs) for the task of translating natural language instructions into effective Linux/Penetration Testing terminal commands.

The project implements a two-stage training pipeline (SFT + GRPO) powered by [Unsloth](https://github.com/unslothai/unsloth) for efficiency and [SwanLab](https://swanlab.cn/) for experiment tracking. It also includes a robust evaluation script supporting multiple models and datasets.

### ✨ Key Features

*   **Stage 1: Supervised Fine-Tuning (SFT):** Trains the model to understand the task and generate Chain-of-Thought (CoT) reasoning using `<think>` tags.
*   **Stage 2: Group Relative Policy Optimization (GRPO):** Aligns the model using Reinforcement Learning. Custom reward functions penalize incorrect formats and reward accurate command generation (Syntax & Execution correctness simulation).
*   **Efficient Training:** Utilizes **Unsloth** for faster training and lower memory usage (LoRA/QLoRA).
*   **Concurrent Evaluation:** A multi-threaded evaluator compatible with OpenAI-API format (e.g., vLLM, SGLang) to benchmark various models against standard datasets (NL2SH, Implicit/Explicit Tools).

### 📂 Project Structure

```text
.
├── config.json           # Model configurations for evaluation
├── evaluator.py          # Main evaluation script (Concurrent & OpenAI-API compatible)
├── sft_stage1.py         # Stage 1: SFT Training script
├── grpo_stage2.py        # Stage 2: GRPO/RL Training script
├── .env                  # Environment variables for evaluation
├── training_data/        # Directory for training data
│   ├── sft_think.jsonl   # Data for Stage 1
│   └── grpo_train.jsonl  # Data for Stage 2
└── model/                # Directory for saved models
```

### 🚀 Getting Started

#### 1. Prerequisites

*   Python 3.10+
*   PyTorch (CUDA supported)
*   Unsloth
*   TRL, Transformers, Datasets
*   SwanLab (for logging)

```bash
pip install unsloth "unsloth[colab-new]" @ git+https://github.com/unslothai/unsloth.git
pip install --no-deps trl peft accelerate bitsandbytes
pip install swanlab pandas openai rapidfuzz python-dotenv
```

#### 2. Data Preparation

Prepare your datasets in JSONL format and place them in the `training_data` folder.

*   **SFT Data (`sft_think.jsonl`):**
    ```json
    {"input": "Task description...", "reasoning": "Reasoning steps...", "answer": "ls -la"}
    ```
*   **GRPO Data (`grpo_train.jsonl`):**
    ```json
    {"input": "Task description...", "answer": "ls -la"}
    ```

#### 3. Training

**Stage 1: Supervised Fine-Tuning**

```bash
python sft_stage1.py
```
*   This will finetune a base model (default: `Qwen/Qwen2.5-7B-Instruct`) and save the adapter to `./model/sft/final_model`.

**Stage 2: GRPO Alignment**

> **Note:** Ensure `grpo_stage2.py` points to the correct path of the model trained in Stage 1.

```bash
python grpo_stage2.py
```
*   This loads the SFT model and optimizes it using reward functions (Format Reward + Accuracy Reward).

#### 4. Evaluation

The evaluator works by sending requests to an LLM inference server (like vLLM) that mimics the OpenAI API.

**Configuration:**

1.  Edit `.env` to set your target dataset and API keys.
    ```ini
    DATASET=all_qa  # or 'westen', 'Implicit', 'Explicit'
    OUTPUT_DIR=results
    OPENAI_BASE_URL=http://localhost:8000/v1
    OPENAI_API_KEY=EMPTY
    MAX_WORKERS=10  # Adjust based on your API throughput
    ```
2.  Edit `config.json` to define the models you want to evaluate.

**Run Evaluation:**

```bash
python evaluator.py
```

The script calculates:
*   **Exact Match**
*   **Keyword Recall**
*   **Jaccard Similarity**
*   **Levenshtein Similarity**

Results are saved as CSV files in the `results/` directory.

---

<a name="chinese"></a>

## 🇨🇳 中文说明

**Penetron** 旨在将自然语言指令转化为有效的 Linux 或渗透测试终端命令。

该项目实现了一个包含两个阶段的训练流程（SFT + GRPO），利用 [Unsloth](https://github.com/unslothai/unsloth) 进行高效训练，并使用 [SwanLab](https://swanlab.cn/) 进行实验跟踪。此外，它还包含一个支持多模型对比的强大评估脚本。

### ✨ 主要特性

*   **第一阶段：监督微调 (SFT):** 训练模型理解任务并利用 `<think>` 标签生成思维链（CoT）推理过程。
*   **第二阶段：群组相对策略优化 (GRPO):** 使用强化学习对模型进行对齐。内置自定义奖励函数，用于惩罚错误格式并奖励准确的命令生成（基于标准化命令的匹配度）。
*   **高效训练:** 利用 **Unsloth** 实现更快的训练速度和更低的显存占用 (支持 LoRA/QLoRA)。
*   **并发评估:** 基于 OpenAI-API 格式（兼容 vLLM, SGLang 等）的多线程评估器，支持在标准数据集（NL2SH, Implicit/Explicit Tools）上对多个模型进行基准测试。

### 📂 项目结构

```text
.
├── config.json           # 评估用的模型配置文件
├── evaluator.py          # 主评估脚本（支持并发 & OpenAI-API）
├── sft_stage1.py         # 第一阶段：SFT 训练脚本
├── grpo_stage2.py        # 第二阶段：GRPO/RL 训练脚本
├── .env                  # 评估用的环境变量
├── training_data/        # 训练数据目录
│   ├── sft_think.jsonl   # SFT 阶段数据
│   └── grpo_train.jsonl  # GRPO 阶段数据
└── model/                # 模型保存目录
```

### 🚀 快速开始

#### 1. 环境依赖

*   Python 3.10+
*   PyTorch (支持 CUDA)
*   Unsloth
*   TRL, Transformers, Datasets
*   SwanLab (用于日志记录)

```bash
pip install unsloth "unsloth[colab-new]" @ git+https://github.com/unslothai/unsloth.git
pip install --no-deps trl peft accelerate bitsandbytes
pip install swanlab pandas openai rapidfuzz python-dotenv
```

#### 2. 数据准备

请准备 JSONL 格式的数据集并将其放入 `training_data` 文件夹。

*   **SFT 数据 (`sft_think.jsonl`):**
    ```json
    {"input": "任务描述...", "reasoning": "推理步骤...", "answer": "ls -la"}
    ```
*   **GRPO 数据 (`grpo_train.jsonl`):**
    ```json
    {"input": "任务描述...", "answer": "ls -la"}
    ```

#### 3. 训练流程

**第一阶段：监督微调 (SFT)**

```bash
python sft_stage1.py
```
*   该脚本将微调基础模型（默认：`Qwen/Qwen2.5-7B-Instruct`）并将适配器保存到 `./model/sft/final_model`。

**第二阶段：GRPO 对齐**

> **注意:** 请确保 `grpo_stage2.py` 中的模型路径指向第一阶段训练好的模型路径。

```bash
python grpo_stage2.py
```
*   加载 SFT 模型并利用奖励函数（格式奖励 + 准确性奖励）进行优化。

#### 4. 模型评估

评估器通过向模拟 OpenAI API 的推理服务器（如 vLLM）发送请求来工作。

**配置:**

1.  编辑 `.env` 文件设置数据集和 API 密钥。
    ```ini
    DATASET=all_qa  # 可选 'westen', 'Implicit', 'Explicit' 或 'all_qa'
    OUTPUT_DIR=results
    OPENAI_BASE_URL=http://localhost:8000/v1
    OPENAI_API_KEY=EMPTY
    MAX_WORKERS=10  # 根据你的 API 吞吐量调整并发数
    ```
2.  编辑 `config.json` 定义需要评估的模型列表。

**运行评估:**

```bash
python evaluator.py
```

脚本将计算以下指标：
*   **Exact Match (完全匹配)**
*   **Keyword Recall (关键词召回率)**
*   **Jaccard Similarity (Jaccard 相似度)**
*   **Levenshtein Similarity (编辑距离相似度)**

详细结果和汇总报告将以 CSV 格式保存在 `results/` 目录下。
