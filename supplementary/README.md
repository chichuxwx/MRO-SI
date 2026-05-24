# 可选补充材料

本文件夹只整理补充说明。完整数据集、checkpoint、训练日志和模型权重不放入补充材料。

## 建议提交内容

- `README.md`：项目介绍、方法图、结果图、环境配置与运行说明。
- `scripts/run_full_pipeline.sh`：完整 pipeline，包含生成 masked-route 数据、MRO-SI post-training 和 eval。
- `scripts/build_masked_derivations.py`：masked-route 数据生成与泄露检查。
- `scripts/run_mrosi_train_eval.sh`：训练与 checkpoint 评测。
- `mro_si/`：MRO-SI 核心训练代码。
- `eval/evaluate_math.py`：评测入口。
- `requirements.txt`、`accelerate.yaml`：环境与多卡配置。
- `assets/architecture.png`、`assets/main_results.png`、`assets/loss.png`：框架图、主结果图和目标函数图。

## 最小运行命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
BASE_MODEL=/path/to/Qwen3-1.7B \
MASK_GENERATOR_MODEL=/path/to/Qwen3-4B \
MAX_STEPS=200 \
EVAL_VAL_N=12 \
bash scripts/run_full_pipeline.sh
```

## Masked 数据摘录

如果需要展示 masked 数据结果，只截取 2 到 3 条样例，不提交完整数据集：

```bash
mkdir -p supplementary
head -n 3 data/openthoughts_math_30k_masked.jsonl > supplementary/masked_data_excerpt.jsonl
```

摘录中建议保留 `problem`、`solution`、`masked_derivation`、`masked_derivation_passed`、`masked_derivation_source` 这几类字段，方便说明生成质量和过滤逻辑。

## 不建议提交

- 完整 `data/`、`outputs/`、`logs/`、`wandb/` 目录
- 完整 masked 数据集
- checkpoint 或 adapter 权重
- API key、本地模型路径、机器缓存文件
