# 蒸馏微调（Llamba ⟵ Llama‑3.1‑8B‑Instruct）

本目录提供使用教师模型 Llama‑3.1‑8B‑Instruct 对学生模型 Llamba‑8B‑untied‑unaligned 的蒸馏微调实现：
- 对应 MOHAWK 最终阶段：在对齐混合块后，将教师的其余权重（词嵌入、最终 LayerNorm、LM Head、每层输入归一化与 MLP）迁移到学生；
- 在教师监督下最小化学生与教师 logits 分布的交叉熵（知识蒸馏）。

---

## 安装与准备
- Python 3.10+、CUDA 对应的 PyTorch、`transformers`、`datasets` 等。项目根目录执行：
  ```bash
  pip install -r requirements.txt
  ```
- 如需更快推理/训练，建议安装 Flash‑Attention 2（可选）。
 
- 访问 Hugging Face 受限资源时，需配置访问令牌和缓存目录。

## 目录结构
- `train.py`：入口脚本，解析参数、构建数据与执行两个训练阶段（默认使用 `HuggingFaceFW/fineweb-edu` 的 `CC-MAIN-2014-49` 子集）。
- `arguments.py`：命令行参数定义。
- `data.py`：数据集加载与流式打包（按 `seq_length+1` 拼接）。
- `model_utils.py`：教师/学生模型加载、层替换与权重迁移逻辑。
 
- `trainer.py`：蒸馏损失与单阶段训练循环。
- `utils.py`：环境变量、日志、`generate` 补丁与 checkpoint 工具。

### 文件结构分析（Tree + 模块关系）
```
Gather-and-Aggregate/finetune/
├─ README.md                      # 本说明
├─ train.py                       # 训练入口：两阶段蒸馏编排、调度与保存
├─ arguments.py                   # 命令行参数定义
├─ data.py                        # PackedTextDataset 与 DataLoader 构建
├─ model_utils.py                 # 教师/学生加载、层替换、权重迁移、generate 补丁
├─ trainer.py                     # 训练循环、蒸馏损失与日志记录
├─ utils.py                       # 环境变量、日志器、W&B 初始化、保存工具
├─ download_fineweb_subset.py     # FineWeb 子集抽样/分片为 JSONL(.gz)
├─ download_fineweb_all.py        # 镜像下载整个 FineWeb‑Edu 仓库（可筛选子集）
└─ configs/                       # 预留（当前为空）

（模型定义位于仓库上层）
Gather-and-Aggregate/models/llamba.py  # 学生 Llamba 模型实现
```

- 模块关系/数据流（Stage‑1 为例）：
  - `train.py` 读取参数（`arguments.py`）→ 配置环境（`utils.configure_environment`）。
  - 加载教师（HF：Llama‑3.1‑8B‑Instruct）与学生（本仓库 Llamba，见 `models/llamba.py`）（`model_utils.get_*_model`）。
  - 可选：按 `--layers` 用教师对应层替换学生层（`model_utils.apply_replacements`）；
  - 迁移教师的 Embedding / 最终 LayerNorm / LM Head / 每层输入 LN 与 MLP（`model_utils.transfer_teacher_weights`）。
  - 构建数据流（`data.PackedTextDataset`，默认 streaming，将文本拼接为 `seq_length+1`）→ `DataLoader`。
  - 训练循环（`trainer.run_stage`）：
    - 批次驻留 CPU，分别拷到老师/学生设备；
    - 前向老师获取 logits（在学生设备上对齐 dtype）；
    - 前向学生，计算蒸馏损失并反传；
    - AdamW + WSD 调度步进，按间隔日志与保存（`utils.save_checkpoint`）。
  - Stage‑2（可选）：切换到 Open‑Hermes‑2.5，按目标 tokens×epochs 推导步数后复用同一训练循环。

> 备注：`configs/` 目录为未来预置配置占位，当前为空；教师模型从 Hugging Face 动态加载，学生模型定义在 `Gather-and-Aggregate/models/llamba.py`。

## 快速开始
由于目录名包含短横线，不能使用 `python -m ...` 方式运行。请直接执行脚本：
```bash
python Gather-and-Aggregate/finetune/train.py \
  --device cuda:0 \
  --dtype bfloat16 \
  --layers 10 14 17 30 \
  --batch_size 1 \
  --seq_length 4096 \
  --max_steps 1000
```

```bash
nohup setsid env HF_DATASETS_CACHE=/home/liyijia/LinearAttaetion/data/fineweb-cc-main-2014-49 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_API_KEY='e11e417d78f5ac9822301ec6f0e1d0b71d637aa2' WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3,4 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_DEBUG=DETAIL OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4 --master_port=29543 Gather-and-Aggregate/Hybrid_distill/train_gpus.py --dtype bfloat16 --layers 14 16 17 30 --batch_size 2 --epochs 1 --max_steps 10000 --seq_length 1024 --freeze_mlp --freeze_replaced_layers --no_dataset_streaming --local_files_only --save_interval 1000 --skip_stage2 --wandb --wandb_project llamba-distill --wandb_run_name 1teacher_4students1 --output_dir /data/yijia/checkpoints_gpus > logs/train_gpus.log 2>&1 &
```

常用变体：
- 仅做最小验证（更快收敛日志）：`--max_steps 50 --second_dataset_name ""`
- 关闭流式加载（小数据集场景）：`--no_dataset_streaming`
- 离线模式（仅本地已缓存模型/数据）：`--local_files_only`
- 不替换任何层：`--layers -1`（负数将被忽略，相当于空列表）
- 按论文建议冻结 MLP，仅训练混合器：`--freeze_mlp`
- 指定 FineWeb 子集：`--dataset_subset CC-MAIN-2014-49`
- 启用 Weights & Biases 监控：`--wandb --wandb_project <项目名> [--wandb_run_name <自定义名称>]`
- 跳过阶段二：`--skip_stage2`（或 `--second_dataset_name ""`）
 

示例命令（单卡试跑，不保存中间权重）
```bash
CUDA_VISIBLE_DEVICES=7 \
HF_DATASETS_CACHE=/home/liyijia/LinearAttaetion/data/fineweb-cc-main-2014-49 \
python Gather-and-Aggregate/finetune/train.py \
  --device cuda:0 \
  --dtype bfloat16 \
  --layers 10 14 17 30 \
  --batch_size 1 \
  --seq_length 4096 \
  --max_steps 10 \
  --freeze_mlp \
  --no_dataset_streaming \
  --local_files_only \
  --save_interval 1000000 \
  --output_dir /tmp/llamba_trial
```
提示：设置 `CUDA_VISIBLE_DEVICES=7` 后，脚本内 `cuda:0` 即映射到物理 GPU7。若使用 `--epochs`，未提供 `--epoch_step_hint` 时脚本会自动抽样估算每个 epoch 的优化步数，并在日志中显示 ETA。

## 参数总览（节选）
- `--layers`：用教师对应层替换的学生层索引（空/负数表示不替换）。
- `--seq_length`、`--batch_size`、`--grad_accumulation_steps`：控制显存与吞吐。
- `--max_lr=1e-5`、`--min_lr=1e-8`、`--warmup_ratio=0.1`、`--decay_ratio=0.1`：WSD 学习率调度（使用 pytorch-optimizer 的 get_wsd_schedule）。
- `--max_steps`：阶段一步数。阶段二步数由 `second_dataset_tokens` 与 `second_dataset_epochs` 推导。
- `--epochs`：按 epoch 训练的轮数（>0 时忽略 `--max_steps`，每个 epoch 完整遍历一次 Stage‑1 数据）。
- `--second_dataset_name`（默认 Open‑Hermes‑2.5）：留空可跳过阶段二。
- `--skip_stage2`：显式跳过阶段二蒸馏（与将 `--second_dataset_name` 置空等价）。
- `--temperature`：蒸馏温度（默认 1.0）。
- `--output_dir`：checkpoint 输出目录（`checkpoint-<step>`）。
- `--no_dataset_streaming`、`--local_files_only`：数据加载模式控制。
- `--freeze_mlp`：冻结学生模型中全部 MLP 参数，仅更新混合器及其余组件。
- `--freeze_replaced_layers`：冻结通过 `--layers` 替换到学生中的教师层（保持其参数不训练）。
- `--dataset_subset`：FineWeb 子集名称，默认 `CC-MAIN-2014-49`。
- `--wandb` / `--wandb_project` / `--wandb_run_name` / `--wandb_entity` / `--wandb_mode`：Weights & Biases 监控配置。
 

> 运行 `python Gather-and-Aggregate/finetune/train.py --help` 查看全部参数。

### 常用开关详解
- `--no_dataset_streaming`：关闭 Stage‑1 的 Hugging Face datasets 流式加载（改为一次性读取），适合小数据/本地完整数据更稳定复现；会增加本地 I/O 与加载时间。注意 Stage‑2 在代码中固定 `streaming=True`，此开关仅影响 Stage‑1（FineWeb‑Edu）。
- `--local_files_only`：所有模型/分词器/数据仅从本地缓存读取，不访问网络。若未命中缓存将报错。实现上对模型传入 `local_files_only=True`，对数据设置 `HF_DATASETS_OFFLINE=1`。建议提前设置 `HF_HOME`/`HF_DATASETS_CACHE` 并用下载脚本预热缓存。
- `--log_interval <N>`：每 `N` 个“优化步”（完成一次梯度累积并执行 `optimizer.step()`）记录一条训练日志（loss、lr、耗时与 ETA 等）。例如 `--log_interval 1` 表示每个优化步都打印。
- `--save_interval <M>`：每 `M` 个“全局步”（micro step，循环内自增）保存一次中间 checkpoint。若启用梯度累积，一个优化步包含多个全局步；例如 `--save_interval 1000000` 在常见短跑下等同于不保存中间权重（结尾仍会保存最终模型）。

## 数据下载
- 若直接使用 Hugging Face streaming，可在训练命令中追加 `--no_dataset_streaming` 关闭流式读取，或保持默认从远端按需加载。离线环境建议提前下载数据并设置 `HF_HOME`/`HF_DATASETS_CACHE`。
- 仅抓取 FineWeb 子集为 JSONL/GZ，可使用：
  ```bash
  python Gather-and-Aggregate/finetune/download_fineweb_subset.py \
    --subset CC-MAIN-2014-49 \
    --output_dir /data/fineweb-cc-2014-49 \
    --samples_per_file 50000 \
    --compression gzip
  ```
  该脚本默认启用 streaming，并将文本写入 `shard-*.jsonl(.gz)`，方便在无网络环境下直接用 `--local_files_only` 训练。
- 若需镜像整个 `HuggingFaceFW/fineweb-edu` 仓库（含所有子集/Parquet），可运行：
  ```bash
  python Gather-and-Aggregate/finetune/download_fineweb_all.py \
    --output_dir /data/fineweb-edu-full \
    --hf_home /data/hf-cache \
    --max_workers 8 \
    --only_parquet
  ```
  支持断点续传、筛选子集/文件模式以及 `--dry_run` 预估体积，下载完成后可设置 `HF_HOME` 指向同一目录以复用缓存。

## 完全执行脚本（含 W&B API Key）
以下命令行在同一终端中先导出 W&B 的 API Key，再执行训练脚本，可直接复制运行：

```bash
export WANDB_API_KEY='e11e417d78f5ac9822301ec6f0e1d0b71d637aa2'
python Gather-and-Aggregate/finetune/train.py \
  --device cuda:0 \
  --teacher_device cuda:1 \
  --dtype bfloat16 \
  --layers 14 16 17 30 \
  --batch_size 1 \
  --epochs 1 \
  --seq_length 1024 \
  --freeze_mlp \
  --no_dataset_streaming \
  --local_files_only \
  --skip_stage2 \
  --wandb \
  --wandb_project llamba-distill \
  --wandb_run_name trial-length1024 \
  --output_dir /tmp/llamba_trial
```

提示：为安全起见，不建议在公共仓库或共享环境中明文保存 API Key。可使用 `wandb login` 方式持久登录，或在使用后 `unset WANDB_API_KEY`。

## 训练流程
1. 加载教师（Llama‑3.1‑8B‑Instruct）与学生（Llamba‑8B）模型；将教师的 rotary embedding 绑定至学生；
2. 按 `--layers` 对学生指定层进行“替换为教师对应层”（深拷贝，不共享参数）；
3. 迁移教师剩余组件权重：词嵌入、最终 LayerNorm、LM Head、每层输入归一化与 MLP；
4. 阶段一（FineWeb‑Edu）：在教师监督下进行知识蒸馏优化（AdamW，β1=0.9、β2=0.95、wd=0.1，WSD 调度）；
5. 阶段二（Open‑Hermes‑2.5，可选）：继续蒸馏若干 epoch，以 2 亿 token/epoch、序列长 4096 为例；
6. 按间隔保存 checkpoint，最终导出完整学生模型权重。

## Checkpoint 与恢复
- Checkpoint 位于：`<output_dir>/checkpoint-<step>`，包含模型与分词器权重。
- 目前未内置从 checkpoint 自动恢复优化器状态；
  - 若中断后继续，可重新启动并设置同样参数；
  - 使用 `--resume_step` 只会影响日志与调度步数对齐，并不会加载优化器状态。

## 性能与显存建议
- 默认 `bfloat16`，建议 A100/H100；显存紧张时：
  - 降低 `--seq_length` 或 `--batch_size`，增大 `--grad_accumulation_steps`；
  - 必要时关闭阶段二或缩短 `--max_steps`；
- Flash‑Attention 2 可降低注意力计算开销（可选）。

## 数据集说明
- 阶段一默认 `HuggingFaceFW/fineweb-edu`，字段名 `text`；
- 阶段二默认 `teknium/OpenHermes-2.5`，字段名 `text`；
- 默认使用流式加载（`datasets` streaming），可通过 `--no_dataset_streaming` 关闭；
- 离线环境可用 `--local_files_only` 并提前准备缓存（可设置 `HF_HOME` 或 `HF_DATASETS_CACHE`）。

## 故障排查
- 不能用 `python -m ...`：目录名含 `-`，请直接执行脚本路径。
- OOM：降低 `seq_length`/`batch_size`，增大梯度累积；确认 `dtype=bfloat16`。
- 模型/数据下载失败：检查网络或使用 `--local_files_only`；设置合适的 HF 缓存目录。
- 生成相关警告：本项目对不支持的 `generate` 参数已做过滤处理，可忽略。

## 监控（Weights & Biases）
- 使用 `--wandb` 可将训练过程同步至 W&B 仪表盘，默认项目名 `llamba-distill`，可通过 `--wandb_project`/`--wandb_run_name`/`--wandb_entity` 自行指定。
- 默认以 `online` 模式运行，也可传入 `--wandb_mode offline` 或 `disabled`。
- 记录的指标包括：`train/loss`（按梯度更新平均）、`train/lr`、`train/grad_norm`、`train/tokens`、`train/optimizer_step` 等，并在 `summary` 中附带可训练参数量、冻结参数量、阶段步数等信息。
- 若未安装 W&B，请先运行 `pip install wandb`，否则脚本会提示缺失依赖。
 

## 扩展
- 自定义数据：参考 `data.PackedTextDataset`；
- 自定义蒸馏损失或调度：参考 `trainer.py`。
- 调整层替换策略：修改/传入 `--layers`，或扩展 `models.apply_replacements`；
 

