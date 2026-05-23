from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mro_si.verifier_utils import extract_boxed_answer, grade_answer


def load_eval_rows(dataset_name: str, num_samples: int | None = None) -> list[dict]:
    from datasets import load_dataset

    name = dataset_name.lower()
    if name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif name == "amo-bench":
        dataset = load_dataset("meituan-longcat/AMO-Bench", split="test")
    elif name == "minerva":
        dataset = load_dataset("math-ai/minervamath", split="test")
    elif name == "amc23":
        dataset = load_dataset("math-ai/amc23", split="test")
    elif name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif name == "aime25":
        dataset = load_dataset("yentinglin/aime_2025", split="train", trust_remote_code=True)
    elif name == "hmmt25":
        dataset = load_dataset("MathArena/hmmt_feb_2025", split="train", trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    rows = []
    for idx, example in enumerate(dataset):
        if name == "amo-bench":
            problem = example["prompt"]
            answer = example["answer"]
            problem_id = example.get("question_id", idx)
        elif name in {"aime24", "aime25", "hmmt25"}:
            problem = example["problem"]
            answer = str(example["answer"])
            problem_id = example.get("id", example.get("problem_idx", idx))
        elif name in {"minerva", "amc23"}:
            problem = example["question"]
            answer = example["answer"]
            problem_id = example.get("id", idx)
        else:
            problem = example["problem"]
            solution = example["solution"]
            answer = extract_boxed_answer(solution) or solution
            problem_id = idx
        rows.append({"problem_id": problem_id, "problem": problem, "answer": answer})
    return rows


def load_vllm_model(
    base_model: str,
    checkpoint_dir: str | None,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    enable_thinking: bool,
):
    from transformers import AutoTokenizer
    from vllm import LLM

    if max_model_len is None:
        max_model_len = 40960 if enable_thinking else 32768

    llm_kwargs = {
        "model": base_model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if checkpoint_dir:
        adapter_path = Path(checkpoint_dir) / "adapter_model.safetensors"
        adapter_bin = Path(checkpoint_dir) / "adapter_model.bin"
        if adapter_path.exists() or adapter_bin.exists():
            llm_kwargs.update(
                {
                    "enable_lora": True,
                    "max_lora_rank": 64,
                    "max_loras": 1,
                    "max_cpu_loras": 1,
                }
            )
        else:
            print(f"Warning: no LoRA adapter weights found in {checkpoint_dir}; evaluating base model.")
            checkpoint_dir = None

    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    return llm, tokenizer, checkpoint_dir


def make_lora_request(checkpoint_dir: str | None):
    if not checkpoint_dir:
        return None
    try:
        from vllm.lora.request import LoRARequest
    except Exception as exc:
        print(f"Warning: could not import LoRARequest: {exc}; evaluating without LoRA.")
        return None
    return LoRARequest("checkpoint_lora", 1, checkpoint_dir)


def evaluate(args) -> dict:
    from vllm import SamplingParams

    rows = load_eval_rows(args.dataset, args.num_samples)
    llm, tokenizer, checkpoint_dir = load_vllm_model(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )
    lora_request = make_lora_request(checkpoint_dir)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        presence_penalty=args.presence_penalty,
        n=args.val_n,
    )

    prompts = []
    for row in rows:
        user_message = (
            f"{row['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        )

    print(
        f"Evaluating {args.dataset} with {len(rows)} problems, val_n={args.val_n}, "
        f"checkpoint={checkpoint_dir or 'base'}"
    )
    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    results = []
    pass_at_n = 0
    total_correct = 0
    formatted_count = 0
    total_generations = len(rows) * args.val_n
    majority_vote_correct = 0

    for idx, (row, output) in enumerate(zip(rows, outputs)):
        generations = []
        predictions = []
        correct_flags = []
        formatted_flags = []

        for item in output.outputs:
            text = item.text
            predicted = extract_boxed_answer(text)
            formatted = predicted is not None
            correct = grade_answer(predicted, row["answer"])
            generations.append(text)
            predictions.append(predicted if predicted is not None else "[No boxed answer found]")
            correct_flags.append(correct)
            formatted_flags.append(formatted)

        formatted_predictions = [
            pred for pred, formatted in zip(predictions, formatted_flags) if formatted
        ]
        majority_correct = False
        if formatted_predictions:
            majority_answer = Counter(formatted_predictions).most_common(1)[0][0]
            majority_correct = grade_answer(majority_answer, row["answer"])

        num_correct = sum(correct_flags)
        has_correct = any(correct_flags)
        total_correct += num_correct
        formatted_count += sum(formatted_flags)
        pass_at_n += int(has_correct)
        majority_vote_correct += int(majority_correct)

        results.append(
            {
                "problem_id": row["problem_id"],
                "problem": row["problem"],
                "ground_truth": row["answer"],
                "val_n": args.val_n,
                "generations": [
                    {
                        "predicted_answer": pred,
                        "full_generation": gen,
                        "correct": corr,
                        "formatted": fmt,
                    }
                    for pred, gen, corr, fmt in zip(predictions, generations, correct_flags, formatted_flags)
                ],
                "num_correct": num_correct,
                "pass_at_n": has_correct,
                "majority_vote_correct": majority_correct,
                "predicted_answer": predictions[0],
                "full_generation": generations[0],
                "correct": correct_flags[0],
                "formatted": formatted_flags[0],
            }
        )
        print(
            f"[{idx + 1}/{len(rows)}] pass@{args.val_n}={100 * pass_at_n / (idx + 1):.2f}% "
            f"avg@{args.val_n}={100 * total_correct / ((idx + 1) * args.val_n):.2f}%"
        )

    num_problems = len(rows)
    summary = {
        "base_model": args.base_model,
        "checkpoint_dir": checkpoint_dir,
        "dataset": args.dataset,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "max_new_tokens": args.max_new_tokens,
        "val_n": args.val_n,
        "num_problems": num_problems,
        "total_solutions": total_generations,
        "pass_at_n": pass_at_n,
        "pass_at_n_pct": 100 * pass_at_n / max(1, num_problems),
        "average_at_n": total_correct,
        "average_at_n_pct": 100 * total_correct / max(1, total_generations),
        "majority_vote_at_n": majority_vote_correct,
        "majority_vote_at_n_pct": 100 * majority_vote_correct / max(1, num_problems),
        "formatted_count": formatted_count,
        "format_rate": 100 * formatted_count / max(1, total_generations),
        "results": results,
    }

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"Saved results to {output_path}")

    print(
        f"Final average@{args.val_n}: {summary['average_at_n_pct']:.2f}% | "
        f"pass@{args.val_n}: {summary['pass_at_n_pct']:.2f}% | "
        f"majority@{args.val_n}: {summary['majority_vote_at_n_pct']:.2f}%"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a base model or LoRA checkpoint on math benchmarks.")
    parser.add_argument("--base_model", required=True, help="Base model path or Hugging Face id.")
    parser.add_argument("--checkpoint_dir", default=None, help="Optional LoRA checkpoint directory.")
    parser.add_argument(
        "--dataset",
        default="math500",
        choices=["math500", "amo-bench", "aime24", "aime25", "hmmt25", "minerva", "amc23"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=38912)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--val_n", type=int, default=12)
    args = parser.parse_args()
    if args.top_p is None:
        args.top_p = 0.95 if args.enable_thinking else 0.8
    if args.output_file is None:
        parts = ["eval_results", args.dataset, Path(args.base_model).name]
        if args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir)
            parts += [checkpoint_path.parent.name, checkpoint_path.name]
        parts += ["thinking" if args.enable_thinking else "nonthinking", f"temp{args.temperature}", f"valn{args.val_n}"]
        args.output_file = str(Path("eval_results") / ("_".join(parts) + ".json"))
    return args


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
