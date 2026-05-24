# 可复用实验产物说明

本项目清理版只提交轻量、可复用的补充材料。完整数据集、checkpoint 和长日志不进入仓库。

## 已附带

| 文件 | 内容 | 生成方式 |
| --- | --- | --- |
| `supplementary/masked_data_excerpt.jsonl` | 3 条 masked-route 教师标注样例，保留题目、参考答案、解答摘录、masked derivation 和校准状态 | 从完整 `openthoughts_math_30k_masked.jsonl` 中筛选 `masked_derivation_passed=true` 且 `masked_derivation_source=generated` 的样例，并只保留必要字段 |
| `scripts/build_masked_derivations.py` | 教师标注生成脚本 | 默认读取 `siyanzhao/Openthoughts_math_30k_opsd`，调用本地 vLLM 教师模型生成 masked-route 数据 |
| `mro_si/masked_derivation_utils.py` | masked-route 断言与校准逻辑 | 由数据生成脚本调用，检查 `[OMITTED]`、prompt echo、answer leakage、答案复制等问题 |
| `scripts/run_full_pipeline.sh` | 可运行完整 pipeline | 先生成 masked 数据，再调用训练与评测脚本 |

完整 masked 数据集统计如下：

```text
total rows: 29434
passed rows: 24112
generated passed rows: 7520
fallback passed rows: 16592
failed rows: 5322
```

## 运行时生成但不提交

| 产物 | 默认位置 | 说明 |
| --- | --- | --- |
| masked 数据全集 | `data/openthoughts_math_30k_masked.jsonl` 或本地指定路径 | 文件较大，只提交摘录 |
| 训练日志 | `logs/*train*.log` | 由 `scripts/run_mrosi_train_eval.sh` 自动生成 |
| mask 生成日志 | `logs/full_pipeline_mask_build_*.log` | 由 `scripts/run_full_pipeline.sh` 自动生成 |
| 评测结果 | `eval_results/*.json` | 由评测脚本生成，可用来复现主结果图 |
| checkpoint / adapter | `outputs/mrosi/` | 文件较大，不随补充材料提交 |

清理版没有单独维护额外的 memory snapshot；训练状态快照对应 checkpoint / adapter，按上表由脚本生成但不随补充材料提交。

## 复现命令

生成 masked-route 数据：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/build_masked_derivations.py \
  --dataset_name_or_path siyanzhao/Openthoughts_math_30k_opsd \
  --dataset_split train \
  --generation_backend vllm \
  --model_name_or_path /path/to/Qwen3-4B \
  --output_path data/openthoughts_math_30k_masked.jsonl \
  --tensor_parallel_size 4 \
  --batch_size 16 \
  --max_new_tokens 1024 \
  --num_candidates 3 \
  --resume
```

运行完整训练与评测：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
BASE_MODEL=/path/to/Qwen3-1.7B \
MASK_GENERATOR_MODEL=/path/to/Qwen3-4B \
MAX_STEPS=200 \
EVAL_VAL_N=12 \
bash scripts/run_full_pipeline.sh
```
