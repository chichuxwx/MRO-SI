import argparse
import concurrent.futures
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mro_si.masked_derivation_utils import (
    build_privileged_masked_derivation_prompt,
    calibrate_masked_derivation,
    fallback_masked_derivation_from_reference,
)

JSON_PREFILL = "{"
MASKED_DERIVATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "masked_derivation": {"type": "string"},
        "omitted_result": {"type": "string"},
        "checks": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
    },
    "required": ["masked_derivation", "omitted_result", "checks"],
    "additionalProperties": False,
}


def load_source_dataset(name_or_path: str, split: str):
    from datasets import load_dataset

    if Path(name_or_path).is_file():
        return load_dataset("json", data_files=name_or_path)["train"]
    return load_dataset(name_or_path)[split]


def resolve_column(columns: list[str], preferred: str, fallbacks: list[str]) -> str:
    if preferred in columns:
        return preferred
    for column in fallbacks:
        if column in columns:
            return column
    raise ValueError(f"None of the requested columns exist. preferred={preferred!r}, fallbacks={fallbacks}")


def apply_chat_template(
    tokenizer,
    prompt: str,
    enable_thinking: bool = False,
    assistant_prefill: str = "",
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return f"{prompt}\n{assistant_prefill}" if assistant_prefill else prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{rendered}{assistant_prefill}" if assistant_prefill else rendered


def combine_prefill_and_output(prefill: str, output: str) -> str:
    output = output or ""
    if not prefill:
        return output
    stripped = output.lstrip()
    if stripped.startswith("{") or stripped.startswith("```"):
        return output
    return f"{prefill}{output}"


def make_guided_decoding(schema: dict):
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except Exception:
        return None

    for kwargs in ({"json": schema}, {"json_schema": schema}):
        try:
            return GuidedDecodingParams(**kwargs)
        except TypeError:
            continue
    return None


def make_sampling_params(SamplingParams, args, guided_decoding):
    base_kwargs = dict(
        n=max(1, args.num_candidates),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    if guided_decoding is not None:
        try:
            return SamplingParams(**base_kwargs, guided_decoding=guided_decoding), True
        except TypeError:
            pass
    return SamplingParams(**base_kwargs), False


def mask_secret(value: str | None) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 10:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("API base URL is empty. Set --api_base_url or MASK_API_BASE_URL.")
    return base_url


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_api_ssl_context(args):
    if args.api_insecure_skip_verify:
        return ssl._create_unverified_context()
    if args.api_ca_bundle:
        return ssl.create_default_context(cafile=args.api_ca_bundle)
    return None


def _openai_chat_completion_once(prompt: str, args) -> list[str]:
    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key. Export {args.api_key_env}=<token> or pass --api_key.")

    base_url = normalize_base_url(args.api_base_url or os.getenv("MASK_API_BASE_URL", ""))
    endpoint = f"{base_url}/chat/completions"
    n = max(1, args.num_candidates)
    payload = {
        "model": args.api_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate private masked math route hints. "
                    "Return exactly one valid JSON object and no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
        "n": n,
    }
    if args.api_response_format_json:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": args.api_user_agent,
        },
        method="POST",
    )
    ssl_context = build_api_ssl_context(args)
    with urllib.request.urlopen(request, timeout=args.api_timeout, context=ssl_context) as response:
        body = response.read().decode("utf-8")
    result = json.loads(body)
    choices = result.get("choices") or []
    texts = []
    for choice in choices:
        message = choice.get("message") or {}
        text = message.get("content")
        if text is None:
            text = choice.get("text", "")
        texts.append(text or "")
    return texts or [""]


def openai_chat_completion(prompt: str, args) -> list[str]:
    last_error = None
    total_attempts = max(1, args.api_max_retries + 1)
    for attempt in range(total_attempts):
        try:
            return _openai_chat_completion_once(prompt, args)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code == 403:
                hint = (
                    "HTTP 403 came from the API gateway. Check campus/VPN IP allowlist, "
                    "base URL, model id, and whether the gateway blocks non-browser clients. "
                    "You can also try --api_user_agent 'curl/8.0'."
                )
                last_error = RuntimeError(f"HTTP {exc.code}: {body[:500]} ; {hint}")
            else:
                last_error = RuntimeError(f"HTTP {exc.code}: {body[:500]}")
        except Exception as exc:
            if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
                hint = (
                    "TLS certificate verification failed. Set MASK_API_CA_BUNDLE=/path/to/ca.pem, "
                    "or temporarily set MASK_API_INSECURE_SKIP_VERIFY=1 / pass --api_insecure_skip_verify "
                    "for an internal-network experiment."
                )
                last_error = RuntimeError(f"{exc}; {hint}")
                break
            last_error = exc
        if attempt + 1 < total_attempts:
            sleep_s = min(args.api_retry_max_sleep, args.api_retry_base_sleep * (2**attempt))
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI-compatible API generation failed after {total_attempts} attempts: {last_error}")


def generate_openai_api_batch(prompts: list[str], args) -> list[list[str]]:
    results: list[list[str] | None] = [None] * len(prompts)
    workers = max(1, min(args.api_parallel, len(prompts)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(openai_chat_completion, prompt, args): idx for idx, prompt in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return [item if item is not None else [""] for item in results]


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def count_passed_rows(path: Path) -> int:
    if not path.exists():
        return 0
    passed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("masked_derivation_passed"):
                passed += 1
    return passed


def calibration_score(cal) -> tuple[int, int]:
    reason_score = {
        "passed": 100,
        "missing_omission_marker": 20,
        "too_short": 10,
        "answer_not_omitted": 0,
        "answer_copy": -10,
        "meta_leakage": -15,
        "prompt_echo": -20,
        "too_long": -30,
        "parse_failed": -40,
    }.get(cal.reason, -5)
    return reason_score, len(cal.masked_derivation or "")


def choose_calibration(
    raw_candidates: list[str],
    source: dict,
    solution_column: str,
    answer_column: str | None,
    args,
    allow_fallback: bool = True,
):
    reference_answer = str(source.get(answer_column, "")) if answer_column else ""
    best = None
    best_raw = ""
    for raw_text in raw_candidates:
        cal = calibrate_masked_derivation(
            raw_text,
            reference_solution=source[solution_column],
            reference_answer=reference_answer,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        if cal.passed:
            return cal, raw_text, "generated"
        if best is None or calibration_score(cal) > calibration_score(best):
            best = cal
            best_raw = raw_text

    if allow_fallback and args.fallback_to_redacted_solution:
        fallback_text = fallback_masked_derivation_from_reference(
            source[solution_column],
            reference_answer=reference_answer,
            max_chars=min(args.max_chars, args.fallback_max_chars),
        )
        fallback_raw = json.dumps(
            {
                "masked_derivation": fallback_text,
                "omitted_result": "requested result",
                "checks": ["The route follows the worked solution with the requested result omitted."],
            },
            ensure_ascii=False,
        )
        fallback_cal = calibrate_masked_derivation(
            fallback_raw,
            reference_solution=source[solution_column],
            reference_answer=reference_answer,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        if fallback_cal.passed:
            return fallback_cal, fallback_raw, "fallback_redacted_solution"
        if best is None or calibration_score(fallback_cal) > calibration_score(best):
            best = fallback_cal
            best_raw = fallback_raw

    return best, best_raw, "failed"


def main():
    parser = argparse.ArgumentParser(description="Build gated masked derivations for MRO-SI training.")
    parser.add_argument("--dataset_name_or_path", default="data/train.jsonl")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument(
        "--generation_backend",
        default="vllm",
        choices=["vllm", "openai_api"],
        help="Use local vLLM generation or an OpenAI-compatible /v1/chat/completions API.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="",
        help="Local model path for --generation_backend vllm. Kept optional for API-only generation.",
    )
    parser.add_argument(
        "--api_model",
        default=os.getenv("MASK_API_MODEL", "qwen3-32b"),
        help="Model name sent to the OpenAI-compatible API when --generation_backend openai_api.",
    )
    parser.add_argument(
        "--api_base_url",
        default=os.getenv("MASK_API_BASE_URL", ""),
        help="OpenAI-compatible base URL, e.g. https://api.example.edu/v1.",
    )
    parser.add_argument(
        "--api_key_env",
        default="MASK_API_KEY",
        help="Environment variable that stores the API key. The key is never printed.",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="API key override. Prefer --api_key_env/env vars so the token is not stored in shell history.",
    )
    parser.add_argument("--api_parallel", type=int, default=8)
    parser.add_argument("--api_timeout", type=float, default=120.0)
    parser.add_argument("--api_max_retries", type=int, default=4)
    parser.add_argument("--api_retry_base_sleep", type=float, default=1.0)
    parser.add_argument("--api_retry_max_sleep", type=float, default=20.0)
    parser.add_argument(
        "--api_user_agent",
        default=os.getenv("MASK_API_USER_AGENT", "curl/8.0"),
        help="User-Agent header for the API gateway. Some gateways block Python-urllib.",
    )
    parser.add_argument(
        "--api_ca_bundle",
        default=os.getenv("MASK_API_CA_BUNDLE", ""),
        help="Optional CA bundle path for campus/internal API certificates.",
    )
    parser.add_argument(
        "--api_insecure_skip_verify",
        action=argparse.BooleanOptionalAction,
        default=env_flag("MASK_API_INSECURE_SKIP_VERIFY", False),
        help="Disable TLS certificate verification for the API request. Prefer --api_ca_bundle when possible.",
    )
    parser.add_argument(
        "--api_response_format_json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask the API for JSON-object response_format. Disable if the campus proxy/model rejects it.",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--problem_column", default="problem")
    parser.add_argument("--solution_column", default="solution")
    parser.add_argument("--answer_column", default="Answer")
    parser.add_argument("--masked_derivation_column", default="masked_derivation")
    parser.add_argument("--raw_masked_derivation_column", default="raw_masked_derivation")
    parser.add_argument("--status_column", default="masked_derivation_status")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--distributed_executor_backend",
        default="mp",
        choices=["mp", "ray", "uni"],
        help="vLLM distributed backend. Use mp on a single multi-GPU node to avoid Ray placement-group issues.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min_chars", type=int, default=120)
    parser.add_argument("--max_chars", type=int, default=5000)
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--guided_json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assistant_json_prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback_to_redacted_solution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback_max_chars", type=int, default=1800)
    parser.add_argument(
        "--fail_below_pass_rate",
        type=float,
        default=0.0,
        help="Exit non-zero after writing if the final pass rate is below this value.",
    )
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to output_path and skip rows already present in that JSONL file.",
    )
    args = parser.parse_args()

    from tqdm import tqdm

    dataset = load_source_dataset(args.dataset_name_or_path, args.dataset_split)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    columns = list(dataset.column_names)
    problem_column = resolve_column(columns, args.problem_column, ["problem", "Question", "question", "prompt"])
    solution_column = resolve_column(
        columns,
        args.solution_column,
        ["solution", "COT_Reason", "cot_reason", "rationale", "answer", "Answer"],
    )
    answer_column = args.answer_column if args.answer_column in columns else None

    if args.generation_backend == "vllm":
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path is required when --generation_backend vllm.")
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            distributed_executor_backend=args.distributed_executor_backend,
        )
        guided_decoding = make_guided_decoding(MASKED_DERIVATION_JSON_SCHEMA) if args.guided_json else None
        sampling, guided_enabled = make_sampling_params(SamplingParams, args, guided_decoding)
        assistant_prefill = "" if guided_enabled or not args.assistant_json_prefill else JSON_PREFILL
        if args.guided_json and not guided_enabled:
            print("WARNING: vLLM guided JSON decoding is unavailable; falling back to assistant JSON prefill.")
    else:
        tokenizer = None
        llm = None
        sampling = None
        guided_enabled = False
        assistant_prefill = ""
        if args.use_chat_template:
            print("WARNING: --use_chat_template is ignored for --generation_backend openai_api.")
        print(
            "OpenAI-compatible masked generation: "
            f"base_url={normalize_base_url(args.api_base_url or os.getenv('MASK_API_BASE_URL', ''))}, "
            f"model={args.api_model}, key={mask_secret(args.api_key or os.getenv(args.api_key_env))}, "
            f"parallel={args.api_parallel}, response_format_json={args.api_response_format_json}, "
            f"user_agent={args.api_user_agent!r}, "
            f"ca_bundle={args.api_ca_bundle or '<system>'}, "
            f"insecure_skip_verify={args.api_insecure_skip_verify}"
        )
    print(
        "Masked generation controls: "
        f"backend={args.generation_backend}, guided_json={guided_enabled}, "
        f"assistant_prefill={bool(assistant_prefill)}, "
        f"num_candidates={max(1, args.num_candidates)}, retries={max(0, args.retries)}, "
        f"fallback_to_redacted_solution={args.fallback_to_redacted_solution}"
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume_rows = count_jsonl_rows(output_path) if args.resume else 0
    if resume_rows > len(dataset):
        raise ValueError(
            f"Cannot resume: {output_path} has {resume_rows} rows but dataset has only {len(dataset)} rows."
        )
    passed = count_passed_rows(output_path) if args.resume else 0
    total = resume_rows
    mode = "a" if args.resume and resume_rows > 0 else "w"
    if resume_rows:
        print(f"Resuming {output_path}: skipping first {resume_rows}/{len(dataset)} rows.")
    print(
        f"Building masked derivations: dataset_rows={len(dataset)}, "
        f"resume_rows={resume_rows}, remaining_rows={len(dataset) - resume_rows}, "
        f"batch_size={args.batch_size}"
    )

    with output_path.open(mode, encoding="utf-8") as handle:
        total_batches = math.ceil(len(dataset) / args.batch_size)
        initial_batches = resume_rows // args.batch_size
        batch_starts = range(resume_rows, len(dataset), args.batch_size)
        progress = tqdm(
            batch_starts,
            total=total_batches,
            initial=initial_batches,
            desc="masked derivations",
        )
        for start in progress:
            batch = dataset[start : start + args.batch_size]
            problems = batch[problem_column]
            solutions = batch[solution_column]
            prompts = []
            for problem, solution in zip(problems, solutions):
                prompt = build_privileged_masked_derivation_prompt(problem, solution)
                if args.generation_backend == "vllm" and args.use_chat_template:
                    prompt = apply_chat_template(
                        tokenizer,
                        prompt,
                        enable_thinking=args.enable_thinking,
                        assistant_prefill=assistant_prefill,
                    )
                elif assistant_prefill:
                    prompt = f"{prompt}\n{assistant_prefill}"
                prompts.append(prompt)

            raw_candidates_by_row = [[] for _ in prompts]
            pending = list(range(len(prompts)))
            for attempt in range(max(0, args.retries) + 1):
                if not pending:
                    break
                pending_prompts = [prompts[idx] for idx in pending]
                if args.generation_backend == "openai_api":
                    outputs = generate_openai_api_batch(pending_prompts, args)
                else:
                    outputs = llm.generate(pending_prompts, sampling_params=sampling, use_tqdm=False)
                next_pending = []
                for pending_idx, item in zip(pending, outputs):
                    if args.generation_backend == "openai_api":
                        candidate_texts = [combine_prefill_and_output(assistant_prefill, text) for text in item]
                    else:
                        candidate_texts = [
                            combine_prefill_and_output(assistant_prefill, output.text)
                            for output in item.outputs
                        ]
                    raw_candidates_by_row[pending_idx].extend(candidate_texts or [""])

                    source = dict(dataset[start + pending_idx])
                    cal, _, _ = choose_calibration(
                        raw_candidates_by_row[pending_idx],
                        source,
                        solution_column,
                        answer_column,
                        args,
                        allow_fallback=False,
                    )
                    if cal is None or not cal.passed:
                        next_pending.append(pending_idx)
                pending = next_pending

            for row_offset, raw_candidates in enumerate(raw_candidates_by_row):
                source = dict(dataset[start + row_offset])
                cal, raw_text, source_kind = choose_calibration(
                    raw_candidates,
                    source,
                    solution_column,
                    answer_column,
                    args,
                )
                if cal is None:
                    cal = calibrate_masked_derivation(
                        "",
                        reference_solution=source[solution_column],
                        reference_answer=str(source.get(answer_column, "")) if answer_column else "",
                        min_chars=args.min_chars,
                        max_chars=args.max_chars,
                    )
                source[args.masked_derivation_column] = cal.masked_derivation if cal.passed else ""
                source[args.raw_masked_derivation_column] = raw_text
                source[args.status_column] = "passed_fallback" if cal.passed and source_kind != "generated" else cal.reason
                source["masked_derivation_passed"] = bool(cal.passed)
                source["masked_derivation_source"] = source_kind
                if cal.passed:
                    passed += 1
                total += 1
                handle.write(json.dumps(source, ensure_ascii=False) + "\n")

    pass_rate = passed / max(1, total)
    print(f"Masked derivation pass rate: {passed}/{total} = {pass_rate:.3f}")
    print(f"Wrote {output_path}")
    if pass_rate < args.fail_below_pass_rate:
        raise SystemExit(
            f"Masked derivation pass rate {pass_rate:.3f} is below --fail_below_pass_rate "
            f"{args.fail_below_pass_rate:.3f}"
        )


if __name__ == "__main__":
    main()
