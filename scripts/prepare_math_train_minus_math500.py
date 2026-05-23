#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"Could not find any of columns {candidates}; available={columns}")


def normalize_problem(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\\left|\\right", "", text)
    return text


def extract_boxed_answer(text):
    text = str(text or "")
    idx = text.rfind(r"\boxed")
    if idx < 0:
        return ""
    brace = text.find("{", idx)
    if brace < 0:
        return ""
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : pos].strip()
    return ""


def load_train_dataset(name_or_path, split, configs):
    path = Path(name_or_path)
    if path.is_file():
        return load_dataset("json", data_files=str(path))["train"]

    if configs:
        datasets = []
        for config in configs:
            datasets.append(load_dataset(name_or_path, config, split=split))
        return concatenate_datasets(datasets)

    return load_dataset(name_or_path, split=split)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MATH train data and remove any examples whose problem appears in MATH500."
    )
    parser.add_argument("--train_dataset", default="EleutherAI/hendrycks_math")
    parser.add_argument("--train_split", default="train")
    parser.add_argument(
        "--train_configs",
        default=",".join(MATH_CONFIGS),
        help="Comma-separated configs for EleutherAI/hendrycks_math. Set empty for datasets without configs.",
    )
    parser.add_argument("--math500_dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--math500_split", default="test")
    parser.add_argument("--output_path", default="data/math_train_minus_math500.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit after filtering, useful for smoke tests.")
    args = parser.parse_args()

    configs = [item.strip() for item in args.train_configs.split(",") if item.strip()]
    train = load_train_dataset(args.train_dataset, args.train_split, configs)
    math500 = load_dataset(args.math500_dataset, split=args.math500_split)

    train_problem_col = first_existing(train.column_names, ("problem", "Question", "question", "prompt"))
    train_solution_col = first_existing(train.column_names, ("solution", "answer", "Answer", "final_answer"))
    train_answer_col = next((name for name in ("Answer", "answer", "final_answer") if name in train.column_names), None)
    math500_problem_col = first_existing(math500.column_names, ("problem", "Question", "question", "prompt"))

    heldout = {normalize_problem(item[math500_problem_col]) for item in math500}
    seen = set()
    rows = []
    removed_math500 = 0
    removed_duplicate = 0

    for item in train:
        problem = str(item.get(train_problem_col, "")).strip()
        solution = str(item.get(train_solution_col, "")).strip()
        if not problem or not solution:
            continue

        key = normalize_problem(problem)
        if key in heldout:
            removed_math500 += 1
            continue
        if key in seen:
            removed_duplicate += 1
            continue
        seen.add(key)

        answer = str(item.get(train_answer_col, "")).strip() if train_answer_col else ""
        if not answer:
            answer = extract_boxed_answer(solution)

        row = {
            "problem": problem,
            "solution": solution,
            "Answer": answer,
            "text": f"Problem: {problem}",
            "source_dataset": args.train_dataset,
        }
        for optional in ("level", "type", "subject"):
            if optional in item and item[optional] is not None:
                row[optional] = item[optional]
        rows.append(row)

    if args.limit > 0:
        rows = rows[: args.limit]

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        "Prepared MATH train minus MATH500: "
        f"source_rows={len(train)}, math500_rows={len(math500)}, "
        f"removed_math500={removed_math500}, removed_duplicate={removed_duplicate}, "
        f"written={len(rows)}, output={output}"
    )


if __name__ == "__main__":
    main()
