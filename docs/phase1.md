# 阶段 1：复核 PPNL、准备 Conda 环境与固定模型版本

> 更新日期：2026-11-05（北京时间）。状态：四模型固定版本资源均已准备；T5-small/base与Flan-T5-small/base训练和开发评价全部完成。环境及上游版本保持固定。
> 对应计划：[plan.md](../plan.md) 第 4 节。
> 后续阶段统一写入 `docs/phaseX.md`，并使用本文件的八节结构；空白模板见 [phase_template.md](./phase_template.md)。

## 1. 阶段目标与范围

本阶段完成单目标 PPNL 的源码审阅、工作站与 Python 环境准备、四个基线检查点的版本固定，以及可复现的下载、校验和环境检查入口。

本阶段实际完成了环境安装、镜像下载、源码审阅、短检查和两个真实模型的离线加载验收。阶段 1 未生成正式数据集，严格执行器由阶段 2 实施；没有进行训练、GPU pilot 或批量推理评估。

遵守用户对长时间任务的要求：约 1.13 GB 的两个权重文件由用户通过工作站镜像下载，用户也已执行最终离线加载检查。已有运行清单记录北京时间 2026-10-30 00:07:17 的成功验收；本次仅离线重新计算全部模型文件的大小和哈希，没有再次加载模型或调用 GPU。

## 2. 已确认的配置与决策

### 2.1 用户决策与实际硬件

| 项目 | 已确认配置或实测结果 |
| --- | --- |
| GPU | NVIDIA RTX A6000，单卡；按用户要求使用此设备 |
| 显存 | `nvidia-smi` 报告 49,140 MiB，约 48 GiB；空闲量随其他进程变化 |
| 驱动 | 580.178.04 |
| CUDA 运行时 | PyTorch 随包运行时为 12.1；不是系统驱动支持的最高 CUDA 版本 |
| 计算能力 | 8.6；短检查确认支持 BF16 |
| Conda 程序 | `/home/user/miniconda3/bin/conda`，版本 26.1.1 |
| 项目环境名 | `gridpath` |
| 环境路径 | `/home/user/miniconda3/envs/gridpath` |
| Python | 3.10.22 |
| 随机种子 | 后续实验固定为 42；本阶段未训练 |
| 项目 GPU 预算 | 继续按 20 GPU-hours 规划，不因更换显卡自行扩大预算 |

2026-10-29 用户已明确：**基线先使用不带 Flan 的原始 T5-small、T5-base**。本项目据此执行；当前模型范围和统一运行接口见第7节。课程原说明保留原样，实际执行决策写入项目计划和配置。

### 2.2 固定版本与镜像

| 对象 | 准确 ID 或版本 |
| --- | --- |
| PPNL commit | `5fd0376fc50ae9860bc986085165ebac5675fd44` |
| T5-small | `google-t5/t5-small` |
| small revision | `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` |
| T5-base | `google-t5/t5-base` |
| base revision | `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` |
| Flan-T5-small | `google/flan-t5-small`，revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |
| Flan-T5-base | `google/flan-t5-base`，revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` |
| GitHub 镜像 | `https://ghfast.top/https://github.com/` |
| Hugging Face 镜像 | `https://hf-mirror.com` |
| Python 包镜像 | `https://pypi.tuna.tsinghua.edu.cn/simple` |

两个模型的身份以原始 [T5-small 模型卡](https://huggingface.co/google-t5/t5-small) 和 [T5-base 模型卡](https://huggingface.co/google-t5/t5-base) 为依据。实际文件从镜像获取；模型与 tokenizer 均固定到相应 revision，不在后续运行中自动跟随 `main`。

关键依赖已安装并检查：

| 依赖 | 版本 |
| --- | --- |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.40.1 |
| Accelerate | 0.30.1 |
| Tokenizers | 0.19.1 |
| Hugging Face Hub | 0.23.2 |
| SentencePiece | 0.1.99 |
| Datasets | 2.19.2 |
| PyArrow | 16.1.0 |
| NumPy | 1.26.4 |
| scikit-learn | 1.5.2 |
| Matplotlib | 3.9.2 |
| pytest | 8.3.3 |

选择固定的 Transformers 4.x 接口，避免阶段间漂移；显式固定 Accelerate、PyArrow 和 fsspec，并实际构造 Trainer、导入 Datasets 检查兼容性。初始依赖见 [requirements.txt](../requirements.txt)，实际安装记录见 [requirements-lock.txt](../requirements-lock.txt)。

输入长度统一从 **512 tokens** 起步；动作目标初始上限为 256 tokens，路径最多 72 个动作，登记在 [project.json](../configs/project.json)。这只是后续 pilot 的起点，还没有用真实训练样本验证全部任务的长度覆盖率。

## 3. 实施内容与产出

### 3.1 独立 Conda 环境

先尝试利用本机缓存按精确版本离线创建新环境。部分 CUDA 和数值库归档不完整，普通离线求解仍提出远程获取，因此没有把该方案标为可用。

最终将本机已有 `aiaa3201-seg` 环境**离线克隆**为独立的 `gridpath`，复用 Python 和 CUDA 版 PyTorch，再只在新环境内安装项目依赖。原环境没有被修改。新环境保留部分来源环境的附带依赖；本项目实际使用范围由 `requirements.txt` 和配置明确指定。

已额外移除新环境中继承的 SAM-2 编辑安装引用，避免依赖锁指向其他项目源码或触发无关的 GitHub 原站下载。卸载范围仅为新环境中的包记录、`.pth` 和加载器文件；原环境及其源码未修改。

`manifests/conda-cache-specs.txt` 仅是失败尝试使用的候选构建约束，**不是环境锁，也不是推荐的重新安装命令**。成功环境的 Conda 包记录为 `manifests/conda-explicit.txt`（工作站记录，不随发布包上传）。

### 3.2 上游代码获取和审阅

通过工作站网络与 GitHub 镜像进行浅克隆、部分克隆和稀疏检出，只取回审阅所需的源码与说明，未下载上游整套数据，未执行“生成全部数据”脚本。上游固定在 detached HEAD，源码未修改。

[PPNL 原论文](https://arxiv.org/abs/2310.03249)研究自然语言描述下的网格导航，包含提示和 T5/BART 微调实验。本项目只复用单目标环境与执行语义，并增加课程要求的验证、诊断和修复任务。

已核对论文正文第 2–4 节及附录 A/B：[原文模型与训练设置](https://arxiv.org/html/2310.03249v3#S3)使用 T5/BART 的 base、large，硬件为 4×A100，并有特殊分词处理；本项目的小模型、A6000 和课程预算需要独立 pilot 验证，不将本项目成绩直接称作原表数值的复现。原文的 GPT-4 ReAct 反馈实验也不能等同于本课程的首次错误诊断与完整路径修复。

[原文指标说明](https://arxiv.org/html/2310.03249v3#S2.SS3)和结果表的最优率分母存在解释空间；本项目因此同时报告成功样本中的最优率与全样本最优成功率，明确各自分母。课程新增的双错误 OOD、只使用可达地图以及严格布局隔离，均需在最终报告中注明。

审阅发现如下，后续实现必须据此处理：

| 项目 | 上游实际行为 | 本项目处理决策 |
| --- | --- | --- |
| 6×6 基础障碍分布 | `generate_all_sg_data.sh` 第 4–10 行使用 1–5 个障碍 | 必做 ID/OOD 沿用这一地图分布，主要改变错误注入次数 |
| 原始地图数量 | 1 障碍 36 个，2–5 障碍各 200 个，共 836 个生成目标 | 不直接把原始全量数据当本项目训练配额 |
| 更密地图 | 脚本实际使用 6–11 个障碍，README 写 6–12 | 可选实验以代码和明确配置为准，必做部分不启用 |
| 原始划分 | `place_agent_goals.py` 在部分相同布局上划分不同起终点组合 | 重新按规范化障碍布局分组，不能直接使用原始 train/dev/seen 划分 |
| 去重与复现 | 障碍列表顺序影响字典相等比较；未统一固定随机种子，目录枚举未排序 | 阶段 2/3 使用排序后的布局哈希、固定种子和确定性遍历 |
| 可达性 | 放置起终点后没有保证可达，样本可能带 `Goal not reachable` | 进入本项目监督清单前必须做 BFS 可达性检查 |
| 单目标数据生成依赖 | `generate_samples.py` 顶层也导入 `gurobipy` | 不直接导入或运行其全套入口；单目标适配不引入 Gurobi |
| 评估脚本入口 | `executor-point-sg.py` 有无主函数保护的顶层批处理逻辑 | 不直接 import 整个文件，阶段 1 仅 AST 提取指定函数做有限探查 |

上游文件位于 `third_party/ppnl_upstream/ppnl-spatial-temporal-reasoning/`。重点文件：`data-synthesis/generate_all_sg_data.sh`、`generate_envs.py`、`place_agent_goals.py`、`generate_samples.py` 和 `evaluate/executor-point-sg.py`。

### 3.3 执行器语义及阶段 2 的明确边界

`executor-point-sg.py` 第 6–34 行的 `is_goal` 是本次核对的官方单目标函数：

- 坐标是零基 `(行, 列)`；`up` 行减一、`down` 行加一、`left` 列减一、`right` 列加一。
- 网格值 `0/1/2/3` 分别表示空地、障碍、起点、目标。
- 途中经过目标不会提前停止，剩余动作继续执行；只有完整路径的最终位置在目标上才成功。
- 越界返回 `0`，碰障碍返回 `2`，最终到目标返回 `1`；其他未成功也返回 `0`。
- 未知动作和空 token 被静默当作无操作；函数还会尝试给粘连的方向词补空格，并采用 `split(' ')`，因此原始空白格式会影响行为。

例如在测试网格 `[[2,3,0],[0,1,0],[0,0,0]]`、起点 `(0,0)` 上：

| 路径 | 官方返回 | 含义 |
| --- | ---: | --- |
| `right` | 1 | 最终到目标 |
| `right unknown` | 1 | 未知 token 被忽略 |
| `right left` | 0 | 到目标后离开，最终失败 |
| `right left right` | 1 | 离开后返回目标，成功 |
| `down right` | 2 | 撞到障碍 |
| `left` | 0 | 越界 |
| 空路径 | 0 | 正常起终点不同，未到目标 |

这些返回码不能直接构成课程要求的四类错误标签。阶段 2 将保留合法四方向动作的运动和最终目标语义，显式补充：

1. 固定空白规范化，将解析后的合法 token 用单个空格连接，再进行官方差分检查。
2. 严格的 `invalid_action_token` 检查，不再忽略 `jump` 等未知动作，也不自动把粘连词修成合法序列。
3. 首次失败动作位置和逐步坐标轨迹，分别区分碰撞、越界、非法动作和最终未到目标。
4. 动作从 1 编号；`goal_not_reached` 使用结束事件 `L+1`，空路径结束步为 1；这些是项目补充协议。

**一致性口径：**规范化后全部 token 合法的非空候选须与官方最终判断一致；非法 token 上的不同是公开登记的严格适配。空路径另作边界测试。不能同时宣称“完全保留官方未知 token 的行为”与“严格检测非法动作”。

此外，官方距离函数在目标处返回 1，计的是节点数；`is_optimal` 第 97 行比较去空格后的字符数，不是动作数；A* 优先级累计方式也不宜直接作为精确最短距离。阶段 2 使用 BFS 独立计算动作最短距离与最优性。上游参考路径和来源要保留，发现差异时显式记录，不静默改写真值或把字符长度指标当动作最优性。

### 3.4 模型准备和文件校验

已获取两个模型的配置和 tokenizer，合计 4,367,050 bytes。用户随后完成两个 `model.safetensors` 权重下载，避免额外下载 TensorFlow、Flax、ONNX 和重复 PyTorch 权重。2026-10-30 离线复核重新读取全部 11 个文件，大小、SHA-256 和适用的 Git blob 哈希均通过。

| 模型 | 配置与 tokenizer | 权重大小 | 当前权重状态 |
| --- | --- | ---: | --- |
| T5-small | 已下载并校验 | 242,043,056 bytes | 已下载、校验并成功离线加载 |
| T5-base | 已下载并校验 | 891,646,390 bytes | 已下载、校验并成功离线加载 |
| 合计 | 约 4.37 MB 小文件 | **1,133,689,446 bytes，约 1.13 GB** | 全部完成；含小文件共 1,138,056,496 bytes |

[prepare_models.py](../scripts/prepare_models.py) 只使用配置中的 HF 镜像，根据固定 revision 下载文件；支持 `.part` 断点续传，校验文件长度、权重 LFS SHA-256 和小文件 Git blob 哈希，并记录本地 SHA-256。清单见 [models.json](../manifests/models.json)。完整签名下载 URL 不写入记录，只保存最终存储域名。

### 3.5 文件清单

| 文件 | 用途 |
| --- | --- |
| `configs/project.json` | 实际模型、硬件、Conda、镜像、上游 commit、长度上限和种子 |
| `requirements.txt` | 本项目固定的直接依赖 |
| `requirements-lock.txt` | 实际安装版本记录 |
| `manifests/conda-explicit.txt` | Conda 包构建记录 |
| `manifests/models.json` | 固定 revision、文件大小、哈希与下载状态 |
| `manifests/runtime.json` | Python、包版本、GPU、CUDA 和加载检查状态 |
| `manifests/phase1_resources.json` | 阶段状态、资源登记及待执行命令 |
| `manifests/phase1_recheck.json` | 阶段收尾时逐文件大小与哈希复核，以及已有加载验收记录 |
| `manifests/upstream_probe.json` | 官方执行器有限探查结果与源码哈希 |
| `scripts/prepare_models.py` | 镜像下载、续传与文件校验 |
| `scripts/check_phase1.py` | 环境、CUDA、Trainer 接口、tokenizer 和模型加载检查 |
| `scripts/inspect_upstream.py` | 不执行上游顶层入口的有限语义探查 |
| `docs/phase_template.md` | 后续阶段统一中文文档模板 |
| `logs/phase1/` | 安装、环境、上游探查与短检查日志 |
| `logs/phase1/recheck.log` | 2026-10-30 的离线文件复核日志；不是用户最终检查的原始输出 |

## 4. 执行命令

四个模型的资源准备使用同一个入口：

```bash
cd /home/user/cjx/GridPathPlanning
bash scripts/run_phase4_v1.sh prepare all
```

单模型将`all`替换为`t5_base`、`t5_small`、`flan_t5_base`或`flan_t5_small`。下载使用工作站HF镜像，支持断点续传和文件校验，不启动训练。以下环境安装与验收记录保留其实际执行范围。

### 4.1 已执行的主要操作

以下命令用于解释本次实际操作，**已有环境不必重新克隆或重复安装**：

```bash
cd /home/user/cjx/GridPathPlanning

# 成功环境来自本机已有环境的离线克隆，只新增 gridpath。
/home/user/miniconda3/bin/conda create --offline --name gridpath \
  --clone /home/user/miniconda3/envs/aiaa3201-seg --yes

/home/user/miniconda3/bin/conda run --no-capture-output --name gridpath \
  python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements.txt

# 小文件与源码语义检查已执行。
/home/user/miniconda3/bin/conda run --no-capture-output --name gridpath \
  python scripts/prepare_models.py --metadata-only
/home/user/miniconda3/bin/conda run --no-capture-output --name gridpath \
  python scripts/inspect_upstream.py
/home/user/miniconda3/bin/conda run --no-capture-output --name gridpath \
  python scripts/check_phase1.py
```

上游克隆采用 `git -c http.followRedirects=false clone --depth 1 --filter=blob:none --no-checkout`，地址为第 2 节的镜像；随后稀疏检出 `README`、`requirements`、生成和执行器源码并 checkout 固定 commit。所有实际网络获取都通过工作站自身网络执行。

### 4.2 已由用户完成：下载权重并最终验收

用户已完成本节的下载和最终检查，当前无需重复执行。以下保留对应入口，供后续复现时参考。先在工作站终端进入环境：

```bash
cd /home/user/cjx/GridPathPlanning
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate gridpath
```

然后执行：

```bash
# 约1.13 GB；使用已固定的HF镜像和revision，不下载其他框架权重。
python scripts/prepare_models.py
```

下载已由用户在工作站执行，本次收尾没有重新下载或启动后台任务。下载清单分别记录 small、base 权重传输约 53.242 秒和 211.234 秒，这些是脚本记录的单文件耗时，不是全流程总耗时。中断后重新执行同一条命令即可继续 `.part` 文件；校验失败会退出，不能以不完整文件通过验收。

下载成功后进行最终检查：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/check_phase1.py --require-models
```

该检查校验固定清单中的文件，依次在 **CPU** 加载两个模型，检查参数结构后释放；只做微小 CUDA 算子和 Trainer 构造，不调用 `train()`，不开展数据集评估。当前 `manifests/runtime.json` 已记录两个模型 `weights_loaded=true`，且 `phase1_acceptance=complete`。

用户没有保存最终命令的终端日志，`logs/phase1/final-check.log` 不存在；本阶段以已有 `runtime.json` 的验收记录和新生成的 [phase1_recheck.json](../manifests/phase1_recheck.json)、`logs/phase1/recheck.log`（工作站记录，不随发布包上传） 为证据，不补造原始输出。如以后重新执行并需要保留输出，可额外使用 `tee`。

不想激活环境时，可把 `python ...` 替换为：

```bash
/home/user/miniconda3/bin/conda run --no-capture-output --name gridpath python ...
```

模型准备好以后从本地 `models/t5-small`、`models/t5-base` 加载。正式训练、完整 pilot、数据构造及批量推理命令会在相应阶段提供，本阶段不启动。

## 5. 验证结果

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 工作站 GPU 与驱动 | RTX A6000、驱动可访问 | `manifests/runtime.json` |
| 项目 Python | 独立 `gridpath`，Python 3.10.22 | `manifests/runtime.json` |
| 依赖完整性 | `pip check` 返回 `No broken requirements found.` | `logs/phase1/pip-check.log` |
| 主要模块导入 | Torch、Transformers、Datasets、sklearn、Matplotlib、PyYAML 通过 | `logs/phase1/check.log` |
| CUDA 与 BF16 | 16×16 矩阵短算子检查通过 | `manifests/runtime.json` |
| Trainer 接口 | 极小随机 T5 的 CPU Trainer 构造成功；未训练 | `training_api_constructed_without_training=true` |
| small tokenizer | 本地加载成功，示例输入 27 tokens | `manifests/runtime.json` |
| base tokenizer | 本地加载成功，示例输入 27 tokens | `manifests/runtime.json` |
| 上游版本 | HEAD 与固定 commit 一致，源码未改 | Git HEAD、`manifests/upstream_probe.json` |
| 官方行为探查 | 九条路径及两个边界检查通过 | `manifests/upstream_probe.json` |
| 全部文件完整性 | 11 个文件全部重新计算大小和哈希通过，含两个真实权重 | `manifests/phase1_recheck.json`、`logs/phase1/recheck.log` |
| 两个真实模型权重加载 | **用户已完成离线 CPU 加载，均为 `weights_loaded=true`** | `manifests/runtime.json`，`phase1_acceptance=complete` |
| small 参数结构 | 60,506,624 个参数，`d_model=512`；记录的 CPU 加载耗时 0.213 秒 | `manifests/runtime.json` |
| base 参数结构 | 222,903,552 个参数，`d_model=768`；记录的 CPU 加载耗时 0.330 秒 | `manifests/runtime.json` |

最终加载验收记录时间为 UTC 2026-10-29 16:07:17，即北京时间 **2026-10-30 00:07:17**。阶段收尾仅重新读取文件并核对已有记录，没有重新运行 GPU 短检查。

长度方面，small tokenizer 声明上限 512；base 的本地文件没有有效的长度上限元数据，读到的是极大占位整数。**这个占位值不代表可处理无限长输入。** 后续代码必须显式读取项目配置的 512 上限，并在 pilot 中检查三任务的完整输入，不能依赖默认自动截断。

上述 27 tokens 只是接口示例，不是数据集长度统计，也不能替代正式 pilot 的吞吐和显存结果。

## 6. 资源使用记录

| 资源 | 本次情况 |
| --- | --- |
| 正式训练 | 0 次 |
| GPU pilot / 批量推理 | 0 次 |
| GPU 短检查 | 共 4 次：最初 3 次加用户最终验收 1 次；只进行微小算子与环境验证，没有模型训练；本次收尾未增加次数 |
| GPU 时间登记 | 保留本阶段保守预算 0.02 GPU-hours；用户最终检查 wall time 为 3.364011 秒，算子段为 0.160406 秒，见 runtime；缺少各次完整日志，未声称精确累计时长 |
| CPU 工作 | 环境检查、依赖安装、源码审阅、tokenizer 与两个真实模型加载、有限官方函数探查，以及全部文件哈希复核；未做正式数据构造 |
| 上游目录占用 | 本次稀疏检出约 356 KiB，未取回全量数据 |
| 模型小文件 | 4,367,050 bytes，磁盘约 4.3 MiB |
| 项目 Conda 环境 | 约 5.8 GiB，包含复用的 CUDA/PyTorch 和来源环境附带依赖 |
| 已下载权重 | 1,133,689,446 bytes；用户已完成，剩余下载量为 0 |
| 模型文件总量 | 1,138,056,496 bytes；11 个文件均通过离线复核 |

环境安装和人工审阅没有逐步完整计时，因此不填写虚构的 CPU 总耗时或安装总流量。GPU 算子耗时不等于整个进程占用时长，预算按保守占用记录，不将毫秒级算子数值当作全部 GPU 用量。

## 7. 问题与后续处理

当前模型范围为T5-base、T5-small、Flan-T5-base、Flan-T5-small，当前不含BART。版本清单按模型家族分别保存，已有实验的来源与哈希不改写。所有模型统一通过`bash scripts/run_phase4_v1.sh prepare <模型键>`下载与校验，`prepare all`准备全部四个模型；使用工作站HF镜像网络，不使用7899代理隧道。训练和评价流程见[阶段四第4节](phase4.md#4-执行命令)。

| 问题 | 已采取的处理 | 剩余影响 |
| --- | --- | --- |
| T5/Flan-T5 名称冲突 | 记录用户明确选择原始 T5，同步更新计划与实际模型 ID | 已解决 |
| 沙箱内无法访问驱动或 DNS | 通过有权限的工作站执行通道检查 GPU 和访问镜像 | 已核实真实设备；用户在工作站终端运行命令即可 |
| 纯缓存 Conda 求解缺少部分归档 | 改为离线克隆现有 CUDA 环境，在独立环境安装依赖 | 新环境包含部分附带包，需以项目直接依赖区分实际使用范围 |
| 镜像对默认 Python 请求标识返回 403 | 下载脚本显式设置请求标识，小文件及用户执行的大权重传输均完成 | 已解决 |
| 小文件下载限时中断 | 已保存并续传，最终全部校验通过 | 已解决 |
| 上游数据划分共享布局 | 已在计划中明确重新按布局分组 | 阶段 2/3 实施，不复用原始划分 |
| 上游忽略非法动作及错误码不完整 | 记录严格适配与官方一致性的边界 | 阶段 2 实现并检查 |
| 上游距离、最优性指标口径不适用 | 决定独立 BFS 动作距离和统一指标 | 阶段 2/3 实现，不将原指标当最终成绩 |
| base tokenizer 缺有效长度元数据 | 显式登记项目输入长度上限 | 后续 pilot 核实长度分布 |
| 最初权重下载与真实加载未完成 | 用户执行下载与离线最终检查，收尾重新校验全部文件 | 已解决，阶段 1 已验收 |
| 用户最终命令未保留终端日志 | 保留已有运行清单，新增独立离线复核记录，不补造缺失日志 | 精确累计运行时间不可追溯；保留保守预算 |

本阶段没有因为发现上游问题就修改其源码，也没有执行多目标流程。后续修改应落在本项目的适配层，保留来源版本、原始路径与审计记录。

## 8. 验收与下一阶段

- [x] RTX A6000 和指定 Conda 已落实，独立环境可用。
- [x] 原始 T5-small/base 的决策、revision 和下载来源明确。
- [x] PPNL commit 已固定，单目标生成与执行器行为已审阅。
- [x] 依赖、CUDA/BF16、Trainer 构造和 tokenizer 短检查通过。
- [x] 配置、tokenizer、文件校验、运行清单和中文文档模板已提供。
- [x] `plan.md` 已同步实际硬件、模型名称和执行器适配边界。
- [x] 用户执行两个模型权重下载，所有文件哈希通过。
- [x] 执行 `check_phase1.py --require-models`，两个权重离线加载成功，状态变为 `complete`。
- [x] 收尾再次离线校验全部 11 个文件，保留复核清单与日志。

**当前结论：阶段 1 已验收，无需用户再执行本阶段命令，可以进入阶段 2。**

下一阶段是实现单目标执行器适配、BFS 和明确的评测语义，说明写入 `docs/phase2.md`，沿用相同的八节结构。数据构造、训练和完整实验仍按计划分阶段推进。
