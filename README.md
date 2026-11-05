# GridPathPlanning：网格路径规划基线

## 1. 项目目标与范围

AIAA4051 Project 6 的单目标网格路径规划实现，参考 [PPNL 论文](https://arxiv.org/abs/2310.03249)及其[上游代码](https://github.com/MohamedAghzal/llms-as-path-planners)。当前完成阶段一至四 v1：数据构造与审计、严格执行器、原始 T5 的生成微调及单次反馈修复。阶段五主体改进和最终测试尚未开展。

当前验证器是确定性执行器，不是模型自验证；当前结果不能代替课程要求中的自验证对比。详细计划见 [plan.md](plan.md)。

## 2. 配置与方法

- 模型：原始 `google-t5/t5-base`、`google-t5/t5-small`，不使用 Flan；固定 revision 和文件哈希见 [模型清单](manifests/models.json)。
- 数据：6×6 可达网格、1–5 个障碍；4000 条训练样本全部用于生成。开发集分为 G100 与 R100，R100 全为受控无效候选。按障碍布局隔离数据划分，详情见 [阶段三](docs/phase3.md)。
- 训练：单卡 RTX A6000，BF16、全参数 AdamW、学习率 `1e-4`、有效 batch 4、seed 42；三遍共 3000 次更新，pilot 计入预算。每遍在 G100 选模，R100 不参与训练或选模。
- 验证：严格执行完整动作序列，返回有效性、首次错误位置及类型；BFS 只用于离线数据和评分，不替模型规划。
- 提示修复：原网格、候选及执行器反馈输入同一个生成模型，输出完整新路径。
- 前缀续写：保留已执行的合法前缀，以其终点作为新起点，使用生成任务模板预测后缀。前缀已到目标时直接结束，单独统计确定性截断贡献。

## 3. 当前结果与解释

以下全部为**开发集**结果，两个模型均选择第 3 遍检查点；主模型为 T5-base。没有将开发成绩称为最终测试成绩。

| 指标 | T5-base | T5-small |
| --- | ---: | ---: |
| G100 直接生成成功率 | 60% | 23% |
| G100 执行器＋提示修复后成功率 | 64% | 24% |
| G100 执行器＋前缀续写后成功率 | 67% | 40% |
| R100 提示修复成功率 | 58% | 8% |
| R100 前缀续写成功率 | 65% | 19% |

R100 前缀成功数分别为 `57＋8`、`11＋8`，其中后一个数是确定性截断，并非神经网络生成后缀的成功。分错误类型、障碍数量、路径长度、最优性、执行轨迹和耗时等详细指标见 [阶段四](docs/phase4.md)、[结果汇总](results/v1/summary_dev.json)、[指标表](results/v1/metrics_dev.csv)和[逐例复核](results/v1/review_dev.json)。

## 4. 目录与发布内容

| 路径 | 内容 |
| --- | --- |
| `src/gridpath/`、`scripts/`、`tests/` | 实现、运行入口及测试 |
| `configs/`、`manifests/` | 配置、固定版本、数据哈希、历史资源与训练来源记录 |
| `data/` | 冻结来源及 v1 数据；测试数据保留，但未用于调参 |
| `results/v1/` | 开发集汇总、400 条模型×样本记录及图表 |
| `checkpoints/v1/` | 轻量训练记录、逐遍评估和 best 元数据；不含权重 |
| `docs/phase1.md` 至 `docs/phase4.md` | 各阶段完整方法、顺序、命令与验收 |

GitHub 发布包不包含预训练模型、微调权重、优化器状态、Conda 环境、上游仓库副本、安装日志、缓存、论文 PDF 和课程说明原文。工作站原始文件不删除。上传选择由 `scripts/package_repository.py` 的白名单固定，操作见 [GitHub 上传说明](docs/github_upload.md)。

## 5. 环境与权重获取

已有工作站使用 `/home/user/miniconda3/bin/conda` 的 `gridpath` 环境，无需重复安装。新机器在工作站自身网络执行以下命令；更换 Conda 安装位置时替换路径。PyTorch CUDA wheel 与项目其他 Python 包分别安装，不通过 GitHub 隧道下载。

```bash
/home/user/miniconda3/bin/conda create -n gridpath python=3.10 pip -y
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate gridpath
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt safetensors==0.8.0 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --no-deps --no-build-isolation -e .
```

完整原环境包记录见 `requirements-lock.txt`，其中包含来源环境附带依赖；实际复现依赖以上述项目依赖为准。GPU 需支持当前 BF16 配置及兼容 CUDA 驱动。

**预训练权重**：由 `scripts/fetch_pinned_models.py` 从 `https://hf-mirror.com` 下载并校验，两个原始权重合计约 1.134 GB，含 tokenizer/config 约 1.138 GB。base revision 为 `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`，small revision 为 `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`。下节给出下载顺序。

**微调权重**：尚未发布公开下载地址，不能仅下载原始 T5 就复现表中成绩。可按下节重新训练；本工作站的当前权重位于 `checkpoints/v1/t5_base/best/` 和 `checkpoints/v1/t5_small/best/`。如需直接分享，应另行发布这两个目录内的推理文件与 `checkpoint.json`，不包含 `training_state.pt`；不要将它们提交普通 Git。本仓库的模型清单与 best 元数据可用于核对版本和哈希。

## 6. 从头复现的逐步命令

发布目录保留历史完成标记与实验绑定，用于核对证据。**先创建新的独立目录再训练**，避免旧选模锁、训练 lineage 或已有结果与新运行冲突；不要删除原工作站记录。

```bash
# 在下载的本项目根目录运行；目标必须尚不存在。
python scripts/create_reproduction.py --output ../GridPathPlanning-repro
cd ../GridPathPlanning-repro
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate gridpath

# 上游源码通过工作站镜像下载；不下载其整套数据。
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  git clone https://ghfast.top/https://github.com/MohamedAghzal/llms-as-path-planners.git \
  third_party/ppnl_upstream
git -C third_party/ppnl_upstream checkout --detach 5fd0376fc50ae9860bc986085165ebac5675fd44

# 固定 revision、断点续传、完整校验；不改写已发布模型清单。
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python scripts/fetch_pinned_models.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 阶段二、三：CPU 检查、数据构造、完整审计。
python -m pytest -q
python scripts/build_data.py
python scripts/audit_data.py

# 阶段四：以下是正式训练/批量推理，由使用者手动启动。
bash scripts/run_phase4_v1.sh train
bash scripts/run_phase4_v1.sh evaluate
python scripts/review_v1.py
```

若在原工作站复用已有预训练资源，创建目录时加 `--reuse-assets`；这只链接 `models/` 与 `third_party/`，之后跳过克隆及下载，不复用微调检查点。若环境此前以 editable 方式安装到别的目录，切换后重新执行 `python -m pip install --no-deps --no-build-isolation -e .`。

`train` 顺序为 preflight → base pilot/三遍训练 → small pilot/三遍训练 → 开发集选模；`evaluate` 完成 G100/R100 两种修复及汇总。图表生成需要系统中文字体；当前使用 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`。更多分步命令与断点说明见 [阶段四第 4 节](docs/phase4.md)。

只有方案与选模规则最终冻结后才执行下面的最终 ID/OOD 测试；当前仓库未报告其结果：

```bash
bash scripts/run_phase4_v1.sh test
```

## 7. 复现边界与资源记录

当前开发样本数量较小，差异不自动表示统计显著。前缀续写包含确定性截断，必须分别报告其贡献。提示修复没有专门修复监督，其成功不证明模型学会读取执行器反馈，需要后续消融。

GPU 总预算仍为 20 GPU-hours，历史消耗见 `manifests/gpu_budget.json`；它包含已发生的探索成本。独立复现目录不复制该运行账本，会记录自身新实验成本，不改变原工作站历史。不同 GPU、驱动或包版本可能产生数值差异，不能保证逐 token 一致。当前尚无最终测试结论或阶段五改进结论。

## 8. 文档入口与后续工作

按 [阶段一](docs/phase1.md) → [阶段二](docs/phase2.md) → [阶段三](docs/phase3.md) → [阶段四](docs/phase4.md) 阅读；[项目计划](plan.md)说明剩余实验，[上传说明](docs/github_upload.md)说明筛选、建仓库及低流量推送。

项目尚未指定开源许可证；论文与上游实现的权利声明以各自原始来源为准。当前发布不自动为这些第三方内容授予新许可。
