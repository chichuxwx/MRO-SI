# 可复用实验产物说明

本项目清理版只提交轻量、可复用的补充材料。完整数据集、checkpoint 和长日志不进入仓库。

## 已附带

| 文件 | 内容 | 生成方式 |
| --- | --- | --- |
| `supplementary/masked_data_excerpt.jsonl` | 3 条 masked-route 教师标注样例，保留题目、参考答案、解答摘录、masked derivation 和校准状态 | 从完整 `openthoughts_math_30k_masked.jsonl` 中筛选 `masked_derivation_passed=true` 且 `masked_derivation_source=generated` 的样例，并只保留必要字段 |
| `supplementary/eval_summary.csv` | 主实验结果摘要 | 从 `e13/figures/e13_existing_results/summary_metrics.csv` 提取必要列并保留两位小数 |
| `supplementary/length_error_bins_all.csv` | 生成长度分桶与错误率统计 | 来自 `e13/figures/e13_existing_results/length_error_bins_all.csv` |
| `supplementary/leakage_check_summary.json` | train/eval 相似度与泄露检查摘要 | 从 `logg/leakage_check_eval_vs_openthoughts_math_30k_masked_20260512.json` 中只保留数据集级统计 |
| `supplementary/mrosi_vs_opsd_aime24_500_selected.csv` | AIME24 val@1 训练步数趋势摘要 | 来自 `早期实验/figures/mrosi_vs_opsd_aime24_500_positive_selected.csv` |
| `supplementary/start_step_ablation.csv` | self-imitation 启动步数消融摘要 | 来自 `早期实验/figures/start_step_result_curve_data.csv` |
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

## 已检查但不直接提交

- `e13/eval_results/*.json`：完整逐题评测输出，体积较大，已压缩为 `eval_summary.csv`。
- `e13/wandb/`：包含大量运行元数据和本地环境信息，不适合作为补充材料直接提交。
- `logg/*.log`：训练和评测日志较长，复现时会由脚本重新生成。
- `outputs/`、checkpoint、adapter 权重：属于模型产物，不放入轻量补充材料。

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
