# Hybrid Distillation (multi‑GPU)

本目录提供 `train_gpus.py` 的多 GPU 蒸馏流程：学生模型（Llamba‑8B）以 DDP 形式运行，老师（Llama‑3.1‑8B‑Instruct）常驻在一张独立 GPU 上。每个训练步：

1. 各 rank 在本地前向学生模型；
2. Rank0 汇总所有学生输入，在老师卡上前向老师；
3. 老师 logits 被切片并广播回各 rank，计算 CE + KL 蒸馏损失；
4. DDP 同步梯度，ZeroRedundancyOptimizer 更新参数。

## 环境准备

- Python 3.10+，以及支持 bfloat16 的 CUDA GPU。
- 安装依赖（在仓库根目录）

  ```bash
  pip install -r requirements.txt
  ```

- 若需要离线训练，应提前把教师模型和数据集缓存到本地，并设置 `HF_HOME` / `HF_DATASETS_CACHE`。

> **注意**：脚本会将老师部署在最后一张可见 GPU；请确保 `CUDA_VISIBLE_DEVICES` 中的最后一张卡仍有足够显存。

## 关键脚本

| 文件 | 说明 |
| ---- | ---- |
| `train_gpus.py` | DDP 蒸馏训练入口（老师 1 卡 + 学生 N 卡） |
| `arguments.py` | CLI 参数定义（仅保留 `train_gpus.py` 需要的选项） |
| `data.py` | `PackedTextDataset`：将原始文本拼接成长度 `seq_length+1` 的 token 序列 |
| `model_utils.py` | 教师/学生加载、层替换、冻结逻辑 |
| `training_utils.py` | WSD 学习率调度、损失函数、DDP 工具 |
| `utils.py` | 环境变量、日志、W&B 初始化与 checkpoint 保存 |

## 训练命令示例

下面的命令与我们当前的生产配置一致，请根据自己的路径 / 端口调整：

```bash
nohup setsid env \
  HF_DATASETS_CACHE=/home/liyijia/LinearAttaetion/data/fineweb-cc-main-2014-49 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_API_KEY='***' \
  WANDB_MODE=offline \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  NCCL_IB_DISABLE=1 \
  NCCL_SHM_DISABLE=1 \
  NCCL_P2P_DISABLE=1 \
  NCCL_ALGO=Ring \
  NCCL_PROTO=Simple \
  NCCL_MIN_NCHANNELS=1 \
  NCCL_MAX_NCHANNELS=1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  TORCH_NCCL_BLOCKING_WAIT=1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  OMP_NUM_THREADS=4 \
torchrun --standalone --nproc_per_node=4 --master_port=29543 \
  Gather-and-Aggregate/Hybrid_distill/train_gpus.py \
  --dtype bfloat16 \
  --layers 14 16 17 30 \
  --batch_size 2 \
  --ce_weight 1.0 \
  --kl_weight 0.1 \
  --epochs 1 \
  --max_steps 100000 \
  --seq_length 1024 \
  --freeze_mlp \
  --freeze_replaced_layers \
  --no_dataset_streaming \
  --local_files_only \
  --save_interval 1000 \
  --wandb \
  --wandb_project llamba-distill \
  --wandb_run_name 1teacher_4students1 \
  --output_dir /data/yijia/checkpoints_klandce1 \
  > logs/train_gpus1.log 2>&1 &
```

### 常用参数

| 参数 | 说明 |
| ---- | ---- |
| `--dtype` | 训练精度，推荐 `bfloat16` |
| `--layers` | 需要用老师权重替换的学生层索引 |
| `--batch_size` | **每个 rank** 的 micro batch size |
| `--grad_accumulation_steps` | 梯度累积步数（可平衡显存） |
| `--seq_length` | 单个样本的序列长度 |
| `--dataset_name` / `--dataset_subset` | HF 数据集与子集，默认 FineWeb‑Edu `CC-MAIN-2014-49` |
| `--no_dataset_streaming` | 关闭 `datasets` streaming（小数据集可用） |
| `--freeze_mlp` / `--freeze_replaced_layers` | 冻结学生中 MLP 或已替换的层 |
| `--kl_weight` / `--ce_weight` / `--temperature` | 蒸馏损失权重与温度 |
| `--max_steps` / `--epochs` | 控制训练长度（两者择一） |
| `--save_interval` | checkpoint 间隔（以 micro step 计） |
| `--wandb` + `--wandb_*` | 启用 W&B 监控 |
| `--resume_step` | 日志/调度步数对齐（手动恢复时使用） |
| `--local_files_only` | 仅使用本地缓存模型 / 数据 |

`arguments.py` 中已移除未被 `train_gpus.py` 使用的阶段二相关参数，如需扩展请新增选项并保持 README 同步。

### 日志说明

- 仅 rank0 会输出 `[train] micro_step ...` 行；loss 为 rank0 的局部值。若需全局平均，可自行在反向前加 `dist.all_reduce`。
- `tokens_accum` 统计的是当前 rank 的有效 token 数，便于粗略估算吞吐量。
- Checkpoint 保存位置为 `<output_dir>/checkpoint-<step>`；同时会保存 tokenizer。

## 评估

`eval_mmlu_hybrid.py` 用于在 MMLU 上验证蒸馏后的模型。示例命令：

```bash
CUDA_VISIBLE_DEVICES=7 \
python Gather-and-Aggregate/eval_mmlu_hybrid.py \
  --which hybrid_ft \
  --device cuda:0 \
  --dtype bfloat16 \
  --layers 14 16 17 30 \
  --ckpt /data/yijia/checkpoints_klandce1/checkpoint-0
```

只需要一张 GPU，`--ckpt` 指向训练产出的 checkpoint 目录。

## 数据与缓存

- **FineWeb‑Edu** 默认使用 streaming；如需本地化，可将子集缓存到 `HF_DATASETS_CACHE`，并加 `--local_files_only`。
- `PackedTextDataset` 会把文本拼接为长度 `seq_length+1` 的 token 序列：前 `seq_length` 个作为输入，最后一个 token 作为标签（自回归范式）。

## 复现与排障提示

- **OOM**：调小 `seq_length` / `batch_size`，或开启梯度累积；确认老师始终单卡运行。
- **KL 出现 NaN**：`compute_distill_loss` 已做 nan 防护；若仍异常，可先把 `--kl_weight` 降到 0，仅训练 CE。
- **W&B**：离线环境下可用 `WANDB_MODE=offline`，训练结束后再同步。
- **DDP 调试**：将 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 加入环境变量，可查看 NCCL 事件。

## 贡献指南

- 添加新参数时，请同步更新 `arguments.py` 与本文档。
- 若扩展新的训练脚本，建议复用 `training_utils.py` 中的通用工具（WSD 调度、蒸馏损失、DDP init/cleanup）。
- PR 请附带说明：修改动机、关键命令、是否验证通过。

欢迎在 Issues 中反馈问题或提出改进建议。祝训练顺利！
