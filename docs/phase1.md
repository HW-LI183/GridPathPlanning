# 阶段 1：复核 PPNL、准备 Conda 环境与固定模型版本

> 更新日期：2026-11-05（北京时间）。状态：四模型固定版本资源均已准备；T5-small/base与Flan-T5-small/base训练和开发评价全部完成。环境及上游版本保持固定。
> 对应计划：[plan.md](../plan.md) 第 4 节。
> 后续阶段统一写入 `docs/phaseX.md`，并使用本文件的八节结构；空白模板见 [phase_template.md](./phase_template.md)。

## 1. 阶段目标与范围

本阶段完成单目标 PPNL 的源码审阅、实验设备与 Python 环境准备、四个基线检查点的版本固定，以及可复现的下载、校验和环境检查入口。

本阶段实际完成了环境安装、镜像下载、源码审阅、短检查和两个真实模型的离线加载验收。阶段 1 未生成正式数据集，严格执行器由阶段 2 实施；没有进行训练、GPU pilot 或批量推理评估。

遵守用户对长时间任务的要求：约 1.13 GB 的两个权重文件由用户通过实验设备镜像下载，用户也已执行最终离线加载检查。已有运行清单记录北京时间 2026-10-30 00:07:17 的成功验收；本次仅离线重新计算全部模型文件的大小和哈希，没有再次加载模型或调用 GPU。

## 2. 已确认的配置与决策

### 2.1 复现环境

参考GPU为RTX A6000；Python3.10、PyTorch2.5.1+CUDA12.1，支持BF16；seed42、20 GPU-hours预算。Conda使用PATH中的`conda`或`GRIDPATH_CONDA`，无需作者的设备路径、环境克隆来源或连接信息。

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

### 3.1 独立Conda环境

按[公开README](../README.md#5-环境与权重)创建干净的`gridpath`环境，安装固定依赖。公开复现不依赖任何已有私人环境或其他项目；本机环境转储不随仓库发布。

### 3.2 上游代码获取和审阅

通过实验设备网络与 GitHub 镜像进行浅克隆、部分克隆和稀疏检出，只取回审阅所需的源码与说明，未下载上游整套数据，未执行“生成全部数据”脚本。上游固定在 detached HEAD，源码未修改。

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

公开配置和依赖见`configs/project.json`、`requirements.txt`和`requirements-lock.txt`；模型来源见`manifests/models.json`与`flan_models.json`。资源状态与文件核验的公开摘要见`phase1_resources.json`、`phase1_recheck.json`；上游语义探查见`upstream_probe.json`。检查入口为`check_phase1.py`、`inspect_upstream.py`。私人环境转储和终端日志不公开。

## 4. 执行命令

先创建独立复现目录；环境、数据、训练与评价的完整顺序见[复现说明](reproduce.md)。公开结果只读保留，不在此目录继续训练或覆盖汇总。

## 5. 验证结果

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 实验设备 GPU 与驱动 | RTX A6000、驱动可访问 | `manifests/runtime.json` |
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

实验参考GPU为RTX A6000；样本暴露、训练更新与成本摘要见`results/training_summary.json`。公开副本不提供设备名称、系统路径、环境转储、空闲磁盘或连接配置。读者独立运行应记录自己的实际成本。

## 7. 问题与后续处理

模型家族、revision和下载哈希以公开清单为准。镜像下载中断可续传；不得改成未固定版本。上游布局隔离、非法词处理、距离口径及安全加载问题已经记录在第3节，并由阶段2的严格执行器和BFS处理。下载与真实模型加载需要读者自行执行，不是查阅公开结果的前置条件。

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
