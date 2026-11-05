# 阶段 2：单目标执行器、BFS 与统一评分规则

> 更新日期：2026-10-30（北京时间）。状态：**已实现并通过短 CPU 检查，阶段 2 已验收。**
> 对应计划：[plan.md](../plan.md) 第 5 节；前置阶段：[phase1.md](./phase1.md)。
> 本文沿用统一的八节结构，后续说明继续写入 `docs/phaseX.md`。

## 1. 阶段目标与范围

本阶段实现可复用的单目标网格执行器、精确 BFS 最短路径和离线评分接口，为下一阶段构造真值、诊断标签和修复目标提供确定性的基础。

已完成：四类错误的首次定位、逐步坐标轨迹、严格动作解析、官方执行器一致性检查、最短距离计算以及生成、验证和修复指标的分母定义。

阶段 1 的两步也已复核通过：全部 11 个模型文件重新计算大小与哈希一致；两个模型的固定 revision、加载成功状态和 `phase1_acceptance=complete` 相互一致。复核记录为 [phase1_recheck.json](../manifests/phase1_recheck.json)，阶段 1 文档和资源状态已更新为完成。

本阶段没有下载、调用 GPU、训练模型、运行正式 pilot 或创建正式训练/测试数据集。下面的随机路径与人工样例只用于有固定上限的短功能检查，不能作为模型实验成绩。

## 2. 已确认的配置与决策

| 项目 | 本阶段采用值 |
| --- | --- |
| Conda | `/home/user/miniconda3/bin/conda` |
| 环境 | `gridpath`；Python 3.10.22 |
| 正式实验硬件 | NVIDIA RTX A6000；本阶段只使用 CPU |
| 后续基线 | 原始 `google-t5/t5-small` 与 `google-t5/t5-base` |
| 随机检查种子 | 42，使用局部随机数生成器 |
| 网格范围 | 正式基准仍为 6×6；单元检查额外覆盖矩形和单行单列边界 |
| 执行协议版本 | `gridpath.execution.v1` |
| 坐标 | 零基 `(row, col)`，即 `(行, 列)` |
| 动作与 BFS 顺序 | `up, down, left, right` |
| 官方版本 | `5fd0376fc50ae9860bc986085165ebac5675fd44` |
| 官方源码 SHA-256 | `375462625e1c133d5b469037c03a18cbc33160a31b2ba88a0da89a127364a6e4` |

协议保留官方完整序列末尾目标判定；严格非法动作检测、空白归一、首次错误步骤与结束事件则属于明确登记的项目适配。数据构造时应把执行协议版本写入清单，后续更改语义须显式登记。

## 3. 实施内容与产出

### 3.1 网格与公共类型

[types.py](../src/gridpath/types.py) 提供不可变 `Grid`、四方向位移和四类 `ErrorReason`。

`Grid(rows, cols, obstacles, start, goal)` 会检查尺寸、坐标类型、边界和起终点障碍冲突，并将障碍坐标归一为不可变集合。支持官方 `0/1/2/3` 矩阵与 `Grid.from_world()`、`Grid.to_world()` 之间转换。

显式坐标只可补充缺失的矩阵标记，或确认已有的唯一标记；不能通过显式参数掩盖重复起点、重复目标或坐标冲突。允许起终点重合用于边界检查，但正式基准仍排除此类环境。

### 3.2 严格执行与首次错误

[executor.py](../src/gridpath/executor.py) 的主要接口为：

```python
execute(grid, plan) -> ExecutionResult
```

`plan` 可为文本或动作文本序列。解析只归一大小写和空白：`" RIGHT\n"` 等价于 `right`；`right,`、`rightleft`、`jump` 都是非法动作，不删除标点、不拆粘连方向词、不自动修复输出。

每一步按下列顺序处理：

1. 检查当前动作是否属于四方向。
2. 计算尝试位置，先检查越界，再检查障碍物。
3. 合法时更新位置，继续执行下一动作。
4. 首次失败立即停止；只有全部动作合法执行完毕后，才检查最终位置。

这个顺序是**逐步检查**，不会先扫描整条路径的未知 token。例如 `down right jump` 在第 2 步已经碰撞时，不能把尚未执行的第 3 步 `jump` 报为首次错误。

途中到达目标也继续执行后缀：到达后离开且不返回会失败，离开后返回可以成功，后缀碰撞或越界仍判失败。可行但非最短路径保留为有效路径。

错误输出固定为：

| 情况 | 标签与步骤 |
| --- | --- |
| 成功 | `valid` |
| 障碍物碰撞 | `invalid | step=t | reason=obstacle_collision` |
| 越界 | `invalid | step=t | reason=out_of_bounds` |
| 非法动作 | `invalid | step=t | reason=invalid_action_token` |
| 所有动作合法但最终未到目标 | `invalid | step=L+1 | reason=goal_not_reached` |

动作从 1 编号。完全空白的路径解析为空序列：正常起终点不同时为结束步 1 的 `goal_not_reached`；起终点重合时，空路径有效。

返回值包含：

| 字段 | 含义 |
| --- | --- |
| `success` | 整条路径最终是否成功 |
| `feasible` | 应执行的动作是否均合法、无碰撞和越界 |
| `first_error_step`、`error_reason` | 首次失败步骤与类型；成功时均为空 |
| `final_position` | 最后一个合法位置；不会把失败尝试当作实际移动 |
| `executed_steps` | 已成功执行的动作数量，不包含失败动作或结束事件 |
| `full_trace` | 每步执行前、尝试、执行后坐标与错误 |
| `actions` | 归一后的完整候选，包含首次失败后未执行的后缀 |
| `verification_label` | 规范化验证标签属性 |

`TraceStep` 中非法 token 没有尝试位置；`goal_not_reached` 使用 `action=None` 的结束事件。因此 `full_trace` 的长度不总等于 `executed_steps`。`to_dict()` 可直接保存为 JSON。

### 3.3 一段具体执行轨迹

对网格 `[[2,3,0],[0,1,0],[0,0,0]]`，候选为 `down right jump`：

| 步骤 | 动作 | 执行前 | 尝试位置 | 执行后 | 结果 |
| --- | --- | --- | --- | --- | --- |
| 1 | `down` | `(0,0)` | `(1,0)` | `(1,0)` | 合法 |
| 2 | `right` | `(1,0)` | `(1,1)` | `(1,0)` | 障碍物碰撞，停止 |

标签为 `invalid | step=2 | reason=obstacle_collision`，`executed_steps=1`。`jump` 仍保留在候选中，但没有执行轨迹。

同一网格仅执行 `down` 时，动作本身合法，`feasible=True`，但在结束事件 `step=2` 报 `goal_not_reached`，`success=False`。可行性描述的是轨迹合法性，不表示任务已成功，也不等同于上游基于剩余可达性的所有指标。

### 3.4 官方函数加载与一致性边界

[official.py](../src/gridpath/official.py) 在核对源码 SHA-256 后，仅从 AST 提取已审阅的 `is_goal` 定义，不 import 整个上游评估脚本，不执行其顶层批处理、文件读取或无关依赖导入。

差分检查只向官方传入**非空、合法、单空格连接**的动作序列，核对官方最终返回码：成功为 `1`，碰撞为 `2`，越界或未到目标为 `0`。官方没有完整的首次错误诊断，因此步骤、失败坐标及 `0` 的细分类别由轨迹与独立边界检查验证，不能声称它们都直接来自官方返回码。

原始文本上的预期差异也已单独记录：

| 原始候选 | 官方行为 | 本项目行为 |
| --- | --- | --- |
| `right jump` | 忽略未知动作，可能判成功 | 第 2 步非法动作 |
| `RIGHT` | 未识别大写动作 | 归一后执行 `right` |
| `right` 后带换行 | 官方 `split(' ')` 可能留下未知 token | 归一空白后执行 `right` |
| `rightrightleft` | 官方可能补空格执行 | 第 1 步非法动作，不自动修词 |

这些适配差异不会计成合法输入上的差分失败，也不会静默改变上游源码。

### 3.5 精确 BFS

[bfs.py](../src/gridpath/bfs.py) 提供：

```python
shortest_path(grid) -> ShortestPath
```

结果为 `reachable`、`distance`、`actions`、`positions`。距离按动作数计算，位置序列包含起点与终点；起终点重合时距离为 0，不可达时距离为 `None`、两个序列为空。

按 `up/down/left/right` 固定邻居顺序，首次发现位置时登记前驱，保证同一地图得到确定性最短路。未调用上游存在口径问题的距离或最优性函数。

BFS 仅用于离线构造、可达性与最优性评估。`execute()` 和指标聚合不会调用 BFS；后续生成与修复推理也不能直接用 BFS 替代语言模型输出。

### 3.6 统一指标与格式错误处理

[metrics.py](../src/gridpath/metrics.py) 提供四个入口：

```python
parse_verification_label(text)
aggregate_path_metrics(results, reference_distances)
aggregate_verification_metrics(truths, predictions)
aggregate_repair_metrics(original_results, final_results)
```

所有输出同时保留分子、分母计数；空分母返回 `None`，保存成 JSON 时为 `null`，不伪造为 0%。

| 指标 | 固定分母与实现口径 |
| --- | --- |
| `success_rate`、`feasibility` | 全部被评分路径 |
| `optimal_success_rate` | 有可达参考的全部路径，失败计 0；正式基准均可达，因此即全样本最优成功率 |
| `conditional_optimal_rate` | 有可达参考的成功路径 |
| `mean_success_optimality_ratio` | 在这些成功路径上平均 `d*/L`；成功空路径且 `d*=L=0` 时取 1 |
| `macro_f1` | 固定 `valid/invalid` 两类 F1 的平均值 |
| 首错定位、错误类型、联合诊断准确率 | 全部真实无效候选，漏检和格式错误均计错 |
| `repair_success_rate` | 全部原本无效候选，不只统计模型检测出来的候选 |
| `valid_destruction_rate` | 全部原本有效候选 |
| `final_success_rate` | 流程处理后的全部候选；未来端到端评估按对应环境清单使用 |

`reference_distance=None` 明确表示**不可达**，不表示忘记填写参考距离：样本保留在成功率和可行率的分母，最优性中排除并单独计数。如果执行已经成功却提供 `None`，会直接报错；成功长度短于参考距离、列表长度不一致等矛盾输入也会报错。

验证解析仅接受完整的 `valid` 或包含正整数步骤及四类原因的标准标签。容许大小写和空白变化；解释文字、多余字段、缺失诊断、非正整数和未知原因均为格式错误。

格式错误不丢样本，也不强行猜成某个类别：真实类别计一次漏判，不给任一预测类别增加误报，并单独记录格式错误率。验证结果额外输出每类 TP/FP/FN/F1，以及四类真实失败的诊断分组。

## 4. 执行命令

以下检查**已经执行通过**，需要时可用同一 Conda 环境复查，均为短 CPU 任务：

```bash
cd /home/user/cjx/GridPathPlanning
mkdir -p logs/phase2

/home/user/miniconda3/bin/conda run --no-capture-output --name gridpath \
  python -m pytest -q --junitxml=logs/phase2/pytest.xml \
  > logs/phase2/pytest.log 2>&1

/home/user/miniconda3/bin/conda run --no-capture-output --name gridpath \
  python scripts/check_phase2.py \
  > logs/phase2/check.log 2>&1
```

pytest 通过 [pyproject.toml](../pyproject.toml) 的 `pythonpath` 设置找到 `src/`；检查脚本自行设置项目源路径，无须重新安装依赖或访问网络。

`check_phase2.py` 默认使用种子 42、512 条短随机候选，并验证对应 BFS 路径。`--random-cases` 只接受 0–4096，防止把短检查误作全量数据构造。输出默认为 `manifests/phase2_checks.json`；`--output` 可另指定保存位置。

在交互式 Python 中调用接口时，可先在项目根目录设置 `PYTHONPATH=src`。接口使用示例：

```python
from gridpath import Grid, execute
from gridpath.bfs import shortest_path

grid = Grid.from_world([[2, 3, 0], [0, 1, 0], [0, 0, 0]])
result = execute(grid, "down right jump")
print(result.verification_label)
# invalid | step=2 | reason=obstacle_collision

reference = shortest_path(grid)  # 仅离线构造/评估使用
print(reference.distance, reference.actions)
# 1 ('right',)
```

本阶段没有待用户启动的长任务。正式数据构造与审计命令在阶段 3 实现后提供，训练和长时评估仍由用户执行。

## 5. 验证结果

最新完整测试为 **86 passed**，无失败。pytest 自身报告测试耗时 0.05 秒；不包括 Conda 启动及人工审阅时间。

| 检查 | 实际结果 |
| --- | --- |
| 执行器 | 25 项，覆盖四边界、四类错误、首次失败、空路径、目标后缀和轨迹 |
| 网格类型 | 21 项，覆盖矩阵转换、重复/冲突标记、非法坐标及起终点重合 |
| BFS | 10 项，含 80 个固定随机小图与独立全边松弛算法的距离对照 |
| 评分 | 28 项，覆盖条件/全样本分母、漏检、格式错误、不可达、零距离及矛盾真值 |
| 上游函数加载 | 2 项，验证不会执行顶层入口，并拒绝源码哈希漂移 |
| 官方差分比较 | **1,030 次，0 次不一致** |
| BFS 路径官方执行与距离回放 | **512 条全部通过** |

1,030 次比较由 6 条手工合法动作候选、512 条 6×6 短随机路径和 512 条 BFS 最短路径组成。官方返回码统计：成功 `1` 共 521 次、失败 `0` 共 389 次、碰撞 `2` 共 120 次；这些是功能检查覆盖情况，不是语言模型成功率。

证据文件：

- `logs/phase2/pytest.log`（工作站记录，不随发布包上传）、`logs/phase2/pytest.xml`（工作站记录，不随发布包上传）：完整单元检查结果。
- `logs/phase2/check.log`（工作站记录，不随发布包上传）：差分检查摘要。
- [phase2_checks.json](../manifests/phase2_checks.json)：源码哈希、随机种子、官方版本、预期差异、轨迹和评分接口样例。
- [phase1_recheck.json](../manifests/phase1_recheck.json)：阶段 1 权重和文件复核证据。

`phase2_checks.json` 中的人工预测和人工修复只用于确认计数接口，没有调用训练后的模型，不可写入正式实验结果表。

## 6. 资源使用记录

| 资源 | 本阶段情况 |
| --- | --- |
| GPU 使用 | **0 GPU-hours**，没有导入 Torch 或初始化 CUDA |
| 下载 | **0**，全部使用本地源码和文件 |
| 训练、正式 pilot、批量模型推理 | **0 次** |
| 正式数据样本新增 | **0 条** |
| 差分检查 CPU 时间 | 最近一次核心检查约 0.036 秒，精确值保存在检查清单 |
| 单元检查 CPU 时间 | 最近一次 pytest 报告 0.05 秒；完整日志已保存 |
| 阶段 1 权重复核 | 仅 CPU 读取并计算已有文件哈希，没有重新下载或加载 GPU 模型 |

Conda 启动、文件读取和人工审阅没有合并成精确 CPU 总耗时，不填写估算冒充实测。项目原有 GPU 预算登记保持不变。

## 7. 问题与后续处理

| 问题或设计边界 | 处理与当前状态 |
| --- | --- |
| 官方忽略未知动作、没有完整四类诊断 | 严格适配已实现；合法规范序列的最终结果与官方一致 |
| 完整序列与首次到目标语义混淆 | 统一采用完整序列末尾判定，并覆盖离开/返回/后缀失败 |
| 错误注入位置不等于执行失败位置 | 由实际回放生成首错标签，后续构造时另存编辑位置 |
| 空路径和未到目标缺少失败动作 | 使用明确的结束事件 `L+1`，保留可行但未成功的区别 |
| 显式坐标可能掩盖原矩阵冲突 | 已拒绝冲突及重复标记，并加入回归检查 |
| 成功路径却提供不可达参考 | 已作为矛盾真值报错，避免污染最优性分母 |
| 解析失败、漏检导致指标分母缩小 | 不丢样本，分母与所有计数显式保存 |
| BFS 可能被误用为模型修复器 | BFS 与执行/评分模块分离，推理时不得以其结果替代模型输出 |
| 正式实验数据尚未构造 | 按阶段 3 的布局隔离、样本配额和双错误审计规则实施 |

官方源码始终保持原样，适配逻辑位于本项目 `src/gridpath/`。`Grid` 和 BFS 支持的额外边界用于软件正确性检查，不代表已经开展 7×7 或更密环境的额外模型实验。

## 8. 验收与下一阶段

- [x] 阶段 1 的权重、固定版本和实际加载状态复核通过。
- [x] 单目标严格执行器、四类原因、首次步骤和完整轨迹已实现。
- [x] 空白归一、非法 token、空路径和目标后缀规则明确。
- [x] 官方函数的安全加载、源码校验和一致性边界有检查证据。
- [x] BFS 动作最短距离、确定性路径和不可达情形通过检查。
- [x] 评分分母、格式错误、有效路径破坏率和不可达处理已实现。
- [x] 86 项测试及 1,030 次官方差分全部通过。
- [x] 全部工作仅用 CPU，无下载、训练或长时间运行。

**阶段 2 已完成。** 下一阶段是按布局隔离构造三任务数据、单错误 ID 与双错误 OOD，并实施样本配额、哈希及编辑日志审计；说明将写入 `docs/phase3.md`。实际运行前会按任务时长提供用户执行命令。
