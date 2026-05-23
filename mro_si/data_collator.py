from __future__ import annotations

import torch


class MROSIDataCollator:
    """Build student prompts and masked-teacher prompts for MRO-SI.

    Student prompt: problem only.
    Teacher prompt: problem plus a masked derivation route whose final result
    has been omitted.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
        masked_derivation_column: str = "masked_derivation",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.masked_derivation_column = masked_derivation_column

        self.tokenizer.padding_side = "right"

    def _student_prompt(self, problem: str) -> str:
        messages = [
            {
                "role": "user",
                "content": f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            }
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.student_thinking,
        )

    def _teacher_prompt(self, problem: str, masked_derivation: str) -> str:
        masked_derivation = (masked_derivation or "").strip()
        if not masked_derivation:
            raise ValueError(f"Missing required column: {self.masked_derivation_column}")

        user_message = (
            f"Problem: {problem}\n\n"
            "Here is a derivation route with the final result omitted:\n"
            f"=== Masked Derivation Begin ===\n{masked_derivation}\n=== Masked Derivation End ===\n"
            "\nUse the route only to understand the mathematical path. "
            "Re-derive the result from the problem statement in your own words. "
            "Do not mention omitted results, reference solutions, private context, scoring, or verifiers.\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.teacher_thinking,
        )

    def _tokenize_prompts(self, prompts: list[str], prefix: str) -> dict[str, torch.Tensor | int]:
        encoded_no_pad = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_prompt_len = max(lengths)
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        if prefix == "student_prompts":
            attention_key = "student_prompt_attention_mask"
            length_key = "student_prompt_length"
            lengths_key = "student_prompt_lengths_per_example"
        elif prefix == "teacher_prompts":
            attention_key = "teacher_prompt_attention_mask"
            length_key = "teacher_prompt_length"
            lengths_key = "teacher_prompt_lengths_per_example"
        else:
            attention_key = f"{prefix}_attention_mask"
            length_key = f"{prefix}_length"
            lengths_key = f"{prefix}_lengths_per_example"
        return {
            prefix: encoded["input_ids"],
            attention_key: encoded["attention_mask"],
            length_key: max_prompt_len,
            lengths_key: torch.tensor(lengths),
        }

    def __call__(self, features):
        student_prompts = []
        teacher_prompts = []
        problems = []
        solutions = []
        masked_sources = []

        for feature in features:
            problem = str(feature["problem"])
            solution = str(feature["solution"])
            masked_derivation = str(feature.get(self.masked_derivation_column, ""))

            problems.append(problem)
            solutions.append(solution)
            masked_sources.append(
                str(
                    feature.get(
                        f"{self.masked_derivation_column}_source",
                        feature.get("masked_derivation_source", feature.get("source", "unknown")),
                    )
                )
            )
            student_prompts.append(self._student_prompt(problem))
            teacher_prompts.append(self._teacher_prompt(problem, masked_derivation))

        student = self._tokenize_prompts(student_prompts, "student_prompts")
        teacher = self._tokenize_prompts(teacher_prompts, "teacher_prompts")

        return {
            **student,
            **teacher,
            "problems": problems,
            "solutions": solutions,
            "masked_sources": masked_sources,
        }
