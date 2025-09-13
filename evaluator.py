import os
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
import shlex
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset as hf_load_dataset, Dataset
from dotenv import load_dotenv
from rapidfuzz import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed

# 加载 .env 文件中的环境变量
load_dotenv()

# --- 微小优化：预编译正则表达式 ---
# 对于在循环中频繁调用的函数，预编译正则表达式可以提升性能
RE_COMMENT = re.compile(r'#.*')
RE_WHITESPACE = re.compile(r'\s+')
RE_COMMAND = re.compile(r'Command:\s*(.*)', re.IGNORECASE | re.DOTALL)
RE_CODE_BLOCK = re.compile(r'```(?:bash|shell)?\n(.*?)\n```', re.DOTALL)

class Config:
    def __init__(self):
        self.output_dir: str = os.environ["OUTPUT_DIR"]
        self.dataset_name: str = os.environ["DATASET"]
        # --- 新增：并发配置 ---
        # 可以通过环境变量设置，或者直接在这里修改
        # 警告：请根据你API的速率限制（Rate Limit）来调整此数值！
        self.max_workers: int = int(os.getenv("MAX_WORKERS", "1"))

        os.makedirs(self.output_dir, exist_ok=True)

        self.openai_config: Dict[str, str] = {
            "api_key": os.getenv("OPENAI_API_KEY", "EMPTY"),
            "base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        }

        with open('config.json') as f:
            self.models_config: Dict[str, Any] = json.load(f)

        self.dataset_schema: Dict[str, Dict[str, Any]] = {
            "westen": {
                "type": "hf", "path": "westenfelder/NL2SH-ALFA", "subset": "test", 
                "split": "train", "input_key": "nl", "answer_keys": ["bash", "bash2"]
            },
            "qa_variant1": {
                "type": "jsonl", "file_path": "qa_variant1.jsonl", 
                "input_key": "input", "answer_keys": ["output"]
            },
            "qa_variant2": {
                "type": "jsonl", "file_path": "qa_variant2.jsonl", 
                "input_key": "input", "answer_keys": ["output"]
            },
            "qa_variant3": {
                "type": "jsonl", "file_path": "qa_variant3.jsonl", 
                "input_key": "input", "answer_keys": ["output"]
            }
        }
        
        self.evaluate_all_qa = self.dataset_name.lower() == "all_qa"
        
        if not self.evaluate_all_qa and self.dataset_name not in self.dataset_schema:
            raise ValueError(f"不支持的数据集: '{self.dataset_name}'")

# --- 优化：将 config 实例化一次，作为全局常量使用 ---
CONFIG = Config()

def load_data(dataset_name: str) -> Union[Dataset, list]:
    """根据配置加载指定数据集"""
    cfg = CONFIG.dataset_schema[dataset_name]
    
    if cfg["type"] == "hf":
        print(f"正在加载HF数据集: {cfg['path']}")
        load_args = {"path": cfg["path"], "split": cfg["split"]}
        if cfg.get("subset"):
            load_args["name"] = cfg["subset"]
        return hf_load_dataset(**load_args)
    
    elif cfg["type"] == "jsonl":
        file_path = cfg["file_path"]
        print(f"正在加载JSONL数据集: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSONL文件不存在: {file_path}")
        
        # 使用 pandas 读取大型 jsonl 文件可能更快，但当前方法对于中小型文件足够
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"跳过无效的JSON行: {line.strip()}")
        print(f"加载完成，共 {len(data)} 条样本")
        return data
    
    else:
        raise ValueError(f"未知的数据集类型: {cfg['type']}")

def generate_with_openai(client: OpenAI, model_config: Dict, messages: List[Dict]) -> Dict:
    """使用OpenAI API生成回答，包含重试逻辑"""
    max_retries = 3
    backoff_factor = 0.5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_config["name"],
                messages=messages,
                # extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
                temperature=model_config.get("temperature", None),
                max_tokens=model_config.get("max_tokens", 512)
            )
            usage = response.usage
            return {
                "generated_text": response.choices[0].message.content.strip(),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "error": None
            }
        except Exception as e:
            print(f"API 调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {"generated_text": "", "error": str(e), "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            time.sleep(backoff_factor * (2 ** attempt))
    
    return {"generated_text": "", "error": "Max retries exceeded", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def generate_answer(client: OpenAI, model_config: Dict, prompt: str) -> Dict:
    """构造prompt并调用生成函数"""
    system_prompt = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""You are a seasoned Linux command-line expert. 
Your primary mission is to generate accurate and effective terminal commands based on the provided task. Adhere strictly to the required output format.

## Task: {prompt}

### Output Format:
Command:"""}
    ]
    return generate_with_openai(client, model_config, messages)

def extract_final_answer(full_text: str) -> str:
    """从模型的完整输出中提取最终的命令"""
    try:
        if "</think>" in full_text:
            command = full_text.split("</think>")[-1].strip()
        else:
            command = full_text.strip()

        command_match = RE_COMMAND.search(command)
        if command_match:
            command = command_match.group(1).strip()
        
        code_block_match = RE_CODE_BLOCK.search(command)
        if code_block_match:
            command = code_block_match.group(1).strip()

        if len(command) > 1 and command.startswith(('`', "'", '"')) and command[0] == command[-1]:
            command = command[1:-1]
        
        return command.strip()
    except Exception as e:
        print(f"提取命令时出错: {e}")
        return full_text

def normalize_command(cmd: str) -> str:
    """标准化命令字符串"""
    if not isinstance(cmd, str): return ""
    try:
        cmd = cmd.strip()
        cmd = RE_COMMENT.sub('', cmd).strip()
        cmd = RE_WHITESPACE.sub(' ', cmd)
        
        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()
            
        if not parts: return ""
            
        command_name = parts[0]
        args = parts[1:]
        
        options, positional = [], []
        processed_args = []
        for arg in args:
            if arg.startswith('-') and not arg.startswith('--') and len(arg) > 2:
                processed_args.extend(['-' + char for char in arg[1:]])
            else:
                processed_args.append(arg)

        i = 0
        while i < len(processed_args):
            arg = processed_args[i]
            if arg.startswith('-'):
                if '=' in arg:
                    opt, val = arg.split('=', 1)
                    options.append((opt, val))
                elif i + 1 < len(processed_args) and not processed_args[i+1].startswith('-'):
                    options.append((arg, processed_args[i+1]))
                    i += 1
                else:
                    options.append((arg, None))
            else:
                positional.append(arg)
            i += 1
            
        sorted_options = sorted(options, key=lambda x: x[0])
        
        normalized_parts = [command_name]
        for opt, val in sorted_options:
            normalized_parts.append(opt)
            if val is not None:
                normalized_parts.append(val)
        
        normalized_parts.extend(positional)
        return ' '.join(normalized_parts)
    
    except Exception:
        return cmd

def calculate_metrics(ground_truths: List[str], generated: str) -> Dict[str, float]:
    """使用标准化命令计算指标"""
    normalized_generated = normalize_command(generated)
    normalized_ground_truths = [normalize_command(gt) for gt in ground_truths]

    exact_match = 1.0 if any(normalized_generated == ngt for ngt in normalized_ground_truths) else 0.0

    best_levenshtein = max(
        (fuzz.ratio(normalized_generated, ngt) / 100.0 for ngt in normalized_ground_truths),
        default=0.0
    )
    
    generated_tokens = set(normalized_generated.split())
    best_keyword_recall = 0.0
    best_jaccard = 0.0

    for ngt in normalized_ground_truths:
        gt_tokens = set(ngt.split())
        if not gt_tokens: continue

        intersection = gt_tokens.intersection(generated_tokens)
        union = gt_tokens.union(generated_tokens)

        best_keyword_recall = max(best_keyword_recall, len(intersection) / len(gt_tokens))
        best_jaccard = max(best_jaccard, len(intersection) / len(union) if union else 0.0)

    return {
        "exact_match": exact_match, "keyword_recall": best_keyword_recall,
        "jaccard_similarity": best_jaccard, "levenshtein_similarity": best_levenshtein
    }

# --- 新增：并发处理单元 ---
def process_sample(
    sample_tuple: tuple,
    client: OpenAI,
    model_id: str,
    model_config: Dict,
    dataset_name: str,
    input_key: str,
    answer_keys: List[str]
) -> Optional[Dict]:
    """
    处理单个样本的函数，用于并发调用。
    它包含原来循环内的所有逻辑：API调用、后处理、指标计算。
    """
    i, sample = sample_tuple
    try:
        input_text = sample[input_key]
        
        ground_truths = []
        for key in answer_keys:
            if key in sample and sample[key]:
                gt_value = sample[key]
                if isinstance(gt_value, list):
                    ground_truths.extend([gt.strip() for gt in gt_value])
                else:
                    ground_truths.append(str(gt_value).strip())
        
        if not ground_truths:
            print(f"警告: 样本 {i} 没有有效的标准答案，已跳过。")
            return None

        gen_result = generate_answer(client, model_config, input_text)
        final_answer = extract_final_answer(gen_result["generated_text"])
        metrics = calculate_metrics(ground_truths, final_answer)

        return {
            "dataset": dataset_name, "sample_id": i, "model": model_id,
            "input": input_text, "ground_truths": " | ".join(ground_truths),
            "full_generated_text": gen_result["generated_text"],
            "generated_answer": final_answer, **metrics,
            "prompt_tokens": gen_result["prompt_tokens"],
            "completion_tokens": gen_result["completion_tokens"],
            "total_tokens": gen_result["total_tokens"], "error": gen_result["error"]
        }
    except Exception as e:
        print(f"处理样本 {i} 时发生严重错误: {e}")
        return None

def evaluate_single_dataset(dataset_name: str):
    """评估单个数据集（已重构为支持并发）"""
    data = load_data(dataset_name)
    all_results = []
    
    dataset_cfg = CONFIG.dataset_schema[dataset_name]
    input_key = dataset_cfg["input_key"]
    answer_keys = dataset_cfg["answer_keys"]

    for model_id, model_config in CONFIG.models_config.items():
        print(f"\n=== 正在评测模型: {model_id} ({model_config['name']}) 在数据集: {dataset_name} ===")
        # --- 优化：Client 在模型级别创建一次，而不是每次调用都创建 ---
        client = OpenAI(**CONFIG.openai_config)
        model_results = []
        
        # --- 核心改动：使用 ThreadPoolExecutor 进行并发处理 ---
        with ThreadPoolExecutor(max_workers=CONFIG.max_workers) as executor:
            # 创建一个 future 任务列表
            futures = [
                executor.submit(
                    process_sample, 
                    sample_tuple, 
                    client, 
                    model_id, 
                    model_config, 
                    dataset_name, 
                    input_key, 
                    answer_keys
                ) 
                for sample_tuple in enumerate(data)
            ]
            
            # 使用 tqdm 显示进度，并在任务完成时获取结果
            progress_bar = tqdm(as_completed(futures), total=len(data), desc=f"评测 {model_id}")
            for future in progress_bar:
                result = future.result()
                if result:
                    model_results.append(result)

        all_results.extend(model_results)
    
    if not all_results:
        print(f"数据集 {dataset_name} 评测未产生任何结果。")
        return None, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(all_results)
    
    csv_path = os.path.join(CONFIG.output_dir, f"{dataset_name}_results_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n{dataset_name} 数据集评测完成。详细结果已保存至: {csv_path}")
    
    # 生成汇总报告
    summary_list = []
    for model_name in results_df["model"].unique():
        model_data = results_df[results_df["model"] == model_name]
        if not model_data.empty:
            summary = {
                "dataset": dataset_name, "model": model_name, "samples": len(model_data),
                "accuracy (exact_match)": model_data["exact_match"].mean(),
                "keyword_recall": model_data["keyword_recall"].mean(),
                "jaccard_similarity": model_data["jaccard_similarity"].mean(),
                "levenshtein_similarity": model_data["levenshtein_similarity"].mean(),
                "avg_prompt_tokens": model_data["prompt_tokens"].mean(),
                "avg_completion_tokens": model_data["completion_tokens"].mean(),
                "total_tokens": model_data["total_tokens"].sum()
            }
            summary_list.append(summary)
    
    summary_df = pd.DataFrame(summary_list)
    summary_path = os.path.join(CONFIG.output_dir, f"{dataset_name}_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"{dataset_name} 汇总报告已保存至: {summary_path}")
    
    return results_df, summary_df

def evaluate_all_qa_datasets():
    """评估所有QA数据集并生成总体性能报告 (逻辑不变)"""
    qa_datasets = ["qa_variant1", "qa_variant2", "qa_variant3"]
    all_results, all_summaries = [], []
    
    for dataset_name in qa_datasets:
        results_df, summary_df = evaluate_single_dataset(dataset_name)
        if results_df is not None: all_results.append(results_df)
        if summary_df is not None: all_summaries.append(summary_df)
    
    if not all_summaries:
        print("所有QA数据集评测未产生任何结果。")
        return
    
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_summaries = pd.concat(all_summaries, ignore_index=True)
    
    # 计算总体性能（所有样本的平均值）
    overall_summary_list = []
    for model_name in combined_results["model"].unique():
        model_data = combined_results[combined_results["model"] == model_name]
        if not model_data.empty:
            overall_summary_list.append({
                "dataset": "ALL_QA", "model": model_name, "samples": len(model_data),
                "accuracy (exact_match)": model_data["exact_match"].mean(),
                "keyword_recall": model_data["keyword_recall"].mean(),
                "jaccard_similarity": model_data["jaccard_similarity"].mean(),
                "levenshtein_similarity": model_data["levenshtein_similarity"].mean(),
                "avg_prompt_tokens": model_data["prompt_tokens"].mean(),
                "avg_completion_tokens": model_data["completion_tokens"].mean(),
                "total_tokens": model_data["total_tokens"].sum()
            })

    # 计算每个模型的平均性能（数据集平均）
    avg_summary_list = []
    for model_name in combined_summaries["model"].unique():
        model_data = combined_summaries[combined_summaries["model"] == model_name]
        if not model_data.empty:
            avg_summary_list.append({
                "dataset": "AVERAGE", "model": model_name, "samples": model_data["samples"].sum(),
                "accuracy (exact_match)": model_data["accuracy (exact_match)"].mean(),
                "keyword_recall": model_data["keyword_recall"].mean(),
                "jaccard_similarity": model_data["jaccard_similarity"].mean(),
                "levenshtein_similarity": model_data["levenshtein_similarity"].mean(),
                "avg_prompt_tokens": model_data["avg_prompt_tokens"].mean(),
                "avg_completion_tokens": model_data["avg_completion_tokens"].mean(),
                "total_tokens": model_data["total_tokens"].sum()
            })

    final_summary_df = pd.concat([
        combined_summaries, 
        pd.DataFrame(overall_summary_list), 
        pd.DataFrame(avg_summary_list)
    ], ignore_index=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(CONFIG.output_dir, f"ALL_QA_summary_{timestamp}.csv")
    final_summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"所有QA数据集汇总报告已保存至: {summary_path}")
    
    print("\n=== 所有QA数据集模型评测结果汇总 ===")
    print(final_summary_df.to_string(index=False, float_format="%.4f"))
    
    combined_path = os.path.join(CONFIG.output_dir, f"ALL_QA_results_{timestamp}.csv")
    combined_results.to_csv(combined_path, index=False, encoding='utf-8-sig')
    print(f"所有QA数据集合并的详细结果已保存至: {combined_path}")

def evaluate():
    """主评估函数"""
    if CONFIG.evaluate_all_qa:
        evaluate_all_qa_datasets()
    else:
        evaluate_single_dataset(CONFIG.dataset_name)

if __name__ == "__main__":
    evaluate()