# GitHub 发布与更新操作说明

## 1. 目标与当前状态

当前项目已完成四模型训练和完整开发评价，发布内容包括800条开发案例、24行指标和T5-base主模型选择。已有本地Git仓库为`/home/user/cjx/GridPathPlanning-github`，当前分支为main；本次使用本地快照更新该仓库，**不需要重新新建GitHub仓库**。

本次整理不连接GitHub、不启用隧道、不提交或推送。以下同步、提交、推送命令由用户执行。项目结构见[目录说明](project_structure.md)，项目方法与复现见[README](../README.md)。

## 2. 上传内容与体积

上传：源码、脚本、测试、配置、中文文档、冻结数据、JSON清单、四模型轻量训练记录/逐遍评价/best元数据，以及开发汇总、800条逐例结果和图表。

不上传：预训练/微调权重、优化器状态、Conda环境、上游仓库、日志、Python缓存、模型推理`dev/cache/`、论文PDF及课程说明原文。推理缓存只是可再生中间产物，逐例`cases/`保留；工作站的权重、缓存与原始实验依据不删除。

本次快照共937个文件，原始体积约15.12 MB，gzip估算约1.48 MB。当前模型目录约2.3 GiB、完整检查点约14 GiB，均不整体提交。发布体积精确值见快照同级的`GridPathPlanning-github-v1.selection.json`，字段为`file_count`、`total_bytes`和`tar_gzip_bytes_estimate`。MB为1,000,000 bytes；gzip仅估算可压缩程度，不等于实际VPN流量，Git还有对象与协议开销。

## 3. 生成并核对发布快照

本次已准备的快照路径为`/home/user/cjx/GridPathPlanning-github-v1`。如果之后修改了源码或结果，生成另一个尚不存在的目录，再将下面路径相应替换；脚本拒绝覆盖已有目录：

```bash
cd /home/user/cjx/GridPathPlanning
python scripts/package_repository.py --output ../GridPathPlanning-github-v1
```

本次快照已存在时直接进入第4节，无需重复生成。白名单排除大权重及推理缓存，单文件达到50 MB时停止。`.selection.json`记录逐文件大小与SHA-256，置于快照之外，不提交自身。

## 4. 更新已有本地Git仓库

所有命令在工作站执行，不需要VPN。此前仓库里有未提交的`docs/github_upload.md`修改，先检查并把已有差异存成补丁，再同步：

```bash
cd /home/user/cjx/GridPathPlanning-github
git status --short --branch
git diff -- docs/github_upload.md
git diff > ../GridPathPlanning-github-before-v1.patch

# 先预览，再应用；不使用--delete，不触碰.git和本地Git配置。
rsync -av --checksum --dry-run ../GridPathPlanning-github-v1/ ./
rsync -av --checksum ../GridPathPlanning-github-v1/ ./
```

确认补丁文件未被覆盖；若此前已有同名补丁，使用新的文件名。补丁保存原本的文档定制，可按需保留到新文档中。同步不会删除不在快照中的本地文件，下面仅从发布仓库移除旧版曾跟踪的推理缓存：

```bash
git rm -r --cached --ignore-unmatch -- \
  results/v1/t5_base/dev/cache \
  results/v1/t5_small/dev/cache \
  results/v1/flan_t5_base/dev/cache \
  results/v1/flan_t5_small/dev/cache
```

该操作只取消`GridPathPlanning-github`中的Git跟踪，副本文件也可保留，不影响原工作目录缓存；其余800条`cases/`、训练证据和数据保留。`rsync`不可用时，可使用`cp -a ../GridPathPlanning-github-v1/. ./`同步，再执行同样的缓存移除和Git检查。

## 5. 核对差异与本地提交

```bash
cd /home/user/cjx/GridPathPlanning-github
git add README.md plan.md .gitignore configs docs src scripts tests data manifests results checkpoints
git diff --cached --stat
git diff --cached --name-only
git status --short
```

确认变更包含四模型结果、README和阶段文档；不应有`model.safetensors`、`training_state.pt`、`models/`、`_states/`或`dev/cache/`的新增文件。缓存删除显示为D是预期变化。

```bash
git commit -m 'Complete four-model baseline development evaluation'
git log -1 --oneline
```

到这里仍是纯本地操作。不要重新`git init`，不要修改已有remote或删除`.git`，也不要重置原实验账本。仓库个人身份与既有HTTPS地址沿用现有配置。

## 6. 按需开启隧道并推送

工作站可以直接访问GitHub时可直接push；需要代理时，仅在本地提交准备完成后开启隧道。此前配置为：**工作站侧7899转发到自己电脑侧7897**。在自己电脑终端替换工作站地址运行：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -R 127.0.0.1:7899:127.0.0.1:7897 user@YOUR_WORKSTATION
```

7897必须是自己电脑实际代理监听端口；若代理在7899，将最后一个端口改为7899。SSH登录使用其他端口时补`-p SSH_PORT`。工作站7899被占用时可换成17899，并同步修改push中的代理端口。保持隧道终端打开。

在工作站发布仓库，按代理支持的类型选择一种方式：

```bash
cd /home/user/cjx/GridPathPlanning-github
# SOCKS5代理：
git -c http.proxy=socks5h://127.0.0.1:7899 push origin main
# 若代理是HTTP/mixed，则使用下行替代：
# git -c http.proxy=http://127.0.0.1:7899 push origin main
```

沿用现有认证即可；若重新提示HTTPS密码，应输入访问令牌，不是GitHub账号密码。令牌不要写入命令、文档或remote URL。推送完成后在浏览器检查README和四模型结果，然后在隧道终端Ctrl+C关闭连接。

如果错误地址被改写成ghproxy.com，核对全局URL重写及现有本仓库规则：

```bash
git config --show-origin --get-regexp '^url\..*\.insteadof$'
git remote -v
```

需要时保留全局配置，在本仓库设置完整地址的精确规则：

```bash
git config --local 'url.https://github.com/JiongxuChen/GridPathPlanning.git.insteadOf' \
  'https://github.com/JiongxuChen/GridPathPlanning.git'
```

其他仓库替换完整URL。镜像用于下载上游源码，不用于写入GitHub仓库；不要为排错去重复克隆整个项目。

## 7. 流量控制与日常维护

先离线完成筛选、同步、差异检查和commit，再开隧道push，Git只需传输新增对象。原版已提交的缓存会保留在旧提交历史中，本次只从当前版本移除，不重写历史或强推。

模型与Python包下载始终使用工作站网络；HF/上游源码按README使用镜像，不通过7899隧道。仅push命令使用`-c http.proxy=...`，不设置全局代理、不上传权重、不反复删仓库重传。原始工作目录和Git发布目录独立，日后结果变更仍需重新生成快照并同步。

工作站已清理可再生Python测试/字节码缓存；训练状态、GPU账本、推理缓存与原始日志保留。实际内容分工见[项目目录说明](project_structure.md)。

## 8. 首次建仓库时的备用步骤

本项目当前无需再建仓库。只有发布到另一个全新仓库时，在GitHub浏览器中选择New repository，填写名称和可见性，不自动添加README、.gitignore或许可证；在一个新的发布目录执行：

```bash
git init -b main
git config user.name '你的GitHub用户名'
git config user.email '你的提交邮箱'
git add .
git commit -m 'Publish GridPathPlanning baseline'
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

然后按第6节完成首次push（添加`-u origin main`）。GitHub官方参考：[上传本地代码](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、[访问令牌](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)。
