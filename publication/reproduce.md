# 复现：从公开结果到独立训练

## 1. 目标与边界

本说明覆盖v1、v2及基于v2的验证改进。发布结果用于查阅与CPU评分复核；重新训练必须在独立目录中进行。正式GPU任务由使用者启动，脚本不会在查阅或打包时自动训练。

## 2. 环境准备

按[README环境说明](../README.md#5-环境与权重)创建Conda `gridpath`环境。运行入口默认调用PATH中的`conda`；需要自定义时设置`GRIDPATH_CONDA`为自己的Conda可执行文件。无需使用作者的目录、用户名或SSH设置。

## 3. 建立独立运行目录与原始资源

在公开仓库根目录执行：

```bash
python scripts/create_public_reproduction.py --output ../GridPathPlanning-run
cd ../GridPathPlanning-run
bash scripts/run_phase4_v1.sh prepare all
```

目标必须在仓库外且不存在。工具只复制源码、通用配置、数据与模型来源清单，不复制已发布结果、检查点、旧选模锁或作者GPU历史。独立运行的初始GPU账本为空，与原实验成本摘要分开。

v1官方一致性审计需要固定上游代码；未下载时使用镜像克隆并核对固定提交：

```bash
git clone https://ghfast.top/https://github.com/MohamedAghzal/llms-as-path-planners.git \
  third_party/ppnl_upstream
git -C third_party/ppnl_upstream checkout 5fd0376fc50ae9860bc986085165ebac5675fd44
```

phase1的版本、模型校验和上游行为审阅见[阶段一](phase1.md)；阶段二的严格执行器、BFS、评分及一致性边界见[阶段二](phase2.md)。资源准备好后，可自行运行`python scripts/inspect_upstream.py`及`python scripts/check_phase2.py`复查。`check_phase1.py --require-models`会执行GPU算子和真实模型加载检查，只在独立运行目录由使用者主动启动；不是查阅结果必需的步骤。

上游源码SHA256另见`manifests/upstream_probe.json`。模型和上游下载使用下载设备自身网络及镜像；GitHub上传代理不参与资源下载。

## 4. v1：生成训练与执行器验证

```bash
python scripts/build_data.py
python scripts/audit_data.py
bash scripts/run_phase4_v1.sh train all
bash scripts/run_phase4_v1.sh evaluate all
```

训练4000G，开发G100/R100，按G100选best。仓库现有v1成绩为开发集；可另运行`bash scripts/run_phase4_v1.sh test all`完成新运行的单错／双错测试，但这不表示原实验已经公布这些测试成绩。

## 5. v2：生成与朴素模型验证

若在同一独立目录中顺序运行v1和v2，首次进入v2前应继承v1成本，避免把总预算归零。**此操作仅在v2尚未开始、其账本仍为空时执行一次：**

```bash
python scripts/create_public_reproduction.py --carry-v1-budget
```

未运行v1则跳过交接。随后：

```bash
bash scripts/run_phase4_v2.sh prepare all
bash scripts/run_phase4_v2.sh train all
bash scripts/run_phase4_v2.sh evaluate all
bash scripts/run_phase4_v2.sh test all
```

G3000/V1000，开发与两个测试各G60/V80/R60。G和V从原始权重独立训练，固定三遍末；`evaluate`自动汇总、dev选模与复核，`test`自动执行两个测试并复核。

## 6. 改进：在v2基础上替换验证器

Part 2主线固定v2开发集选出的T5-base，复用其完整G，不再训练G：

```bash
bash scripts/run_improvement_verify_v1.sh prepare t5_base
bash scripts/run_improvement_verify_v1.sh train t5_base
bash scripts/run_improvement_verify_v1.sh evaluate t5_base
bash scripts/run_improvement_verify_v1.sh test t5_base
bash scripts/run_improvement_verify_v1.sh compare dev t5_base
bash scripts/run_improvement_verify_v1.sh compare test_id t5_base
bash scripts/run_improvement_verify_v1.sh compare test_ood t5_base
```

若从一开始就要复现四模型补充对照，将以上改进入口的`t5_base`全部替换成`all`。不要先完成单模型测试再直接覆盖冻结集合；追加模型需使用已有的受控扩展工具。四模型补充不是必需的Part 2主线。

## 7. 模型选择、续跑与输出

v2和改进入口可用空格指定一个或多个模型键：`t5_base t5_small flan_t5_base flan_t5_small`；`all`表示全部。保持同一流程的比较集合一致，主模型依据dev选择，不根据测试结果改选。v1入口可指定单模型或`all`。

中断后重复原训练命令恢复完整优化器、RNG及样本游标；pilot计入三遍。会话时间、样本暴露和来源不一致时应排查，不能删除进度或GPU账本绕过限制。新结果写入各方法的`results/`，权重写入`checkpoints/`，均只存在于自己的运行目录。

## 8. 复现解释

原实验预测与指标是固定证据；通用路径调整会改变源码指纹，新运行生成自己的指纹，不强行复用原训练身份。完全相同的结果还受硬件、软件版本和确定性设置影响。公开副本的CPU复核只确认已保存预测的评分，不保证重新训练逐条产生相同输出。

一项聚焦消融仍待完成，不能把四模型补充当作该消融。增加数据、轮数或新测试应作为额外工作单独说明。
