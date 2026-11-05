# GitHub 发布：内容筛选与低流量上传

## 1. 目标与范围

将当前阶段一至四 v1 的源码、数据与实验依据发布到新 GitHub 仓库。准备过程完全本地执行；不自动连接 GitHub、不启动隧道、不上传权重、不训练。建议先建 Private 仓库，便于在课程工作完成前控制访问。

本说明针对首次发布。GitHub 的[本地项目上传指南](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)说明了空仓库与首次推送流程；认证参考[个人访问令牌文档](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)。

## 2. 内容选择与体积口径

上传白名单：README、计划、阶段文档、源码、脚本、测试、配置、JSON 清单、冻结数据、v1 开发结果与图表，以及轻量训练记录/逐遍评价/best 元数据。400 条逐例模型评价保留，方便审查结果。

不上传：约 6.4 GiB 的完整检查点目录、约 1.1 GiB 的预训练模型、Conda 环境、上游克隆副本、日志、缓存、论文 PDF、课程说明原文、Conda 本机缓存路径记录。检查点只提取轻量证据，原目录和原权重全部保留。

实际精确大小由打包脚本离线统计，输出 `total_bytes`、`file_count` 和 `tar_gzip_bytes_estimate`。MB 按 1,000,000 bytes，MiB 按 1,048,576 bytes。gzip 数值仅展示可压缩程度，**不是承诺的 VPN 或 Git 传输流量**；实际 push 还有 Git 对象和协议开销。

逐文件大小与 SHA-256 放在发布目录旁的 `.selection.json`，不把它自身加入统计，避免清单自引用。当前已准备的目录及实测大小见本文件末尾。

## 3. 先生成干净发布目录

以下命令不需要网络。目标必须尚不存在；若此前已生成，使用新的目录名，不覆盖旧目录。

```bash
cd /home/user/cjx/GridPathPlanning
python scripts/package_repository.py --output ../GridPathPlanning-github
```

之后只在 `GridPathPlanning-github` 内初始化 Git，不在包含大量运行文件的原工作目录直接 `git add .`。脚本按白名单复制文件；遇到单文件达到 50 MB 会停止，避免意外传入大文件。

原工作目录后续有修改时，重新生成另一个发布目录并核对差异；不要为了同步而把权重目录整体复制过去。

## 4. 新建 GitHub 仓库与准备本地提交

1. 在自己电脑浏览器登录 GitHub，打开 `https://github.com/new`。
2. 仓库名可用 `GridPathPlanning`，填写描述，选择 Private 或自己确定的可见性。
3. **不要勾选自动添加 README、.gitignore 或许可证**，因为本地已有文件；点击 Create repository，复制 HTTPS 地址。
4. 在工作站发布目录执行下列命令。将用户名、邮箱和仓库地址替换为自己的值；邮箱可以使用 GitHub 提供的 noreply 地址。

```bash
cd /home/user/cjx/GridPathPlanning-github
git init -b main
git config user.name '你的 GitHub 用户名'
git config user.email '你的提交邮箱'
git add .
git status --short
git diff --cached --stat
# 确认没有模型、优化器或凭证文件后，创建本地提交。
git commit -m 'Add GridPathPlanning v1 baseline and reproducibility documentation'
git remote add origin https://github.com/YOUR_USERNAME/GridPathPlanning.git
```

以上都是本地操作，不需要打开代理隧道。不要把访问令牌写进 remote URL、脚本或 Git 配置。

## 5. 按需连接 7899 反向隧道

先尝试工作站直连上传；如果可以直连，不需要 VPN：

```bash
git -c http.proxy= push -u origin main
```

若工作站不能连接 GitHub，才启用下面的隧道。前提：**你自己的电脑已经有能访问 GitHub 的代理，其本地监听端口为 7899**；这里的代理端口与工作站 SSH 登录端口不同。

在**自己的电脑**新开终端，替换 SSH 用户、工作站地址及 SSH 端口：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -p YOUR_SSH_PORT \
  -R 127.0.0.1:7899:127.0.0.1:7899 YOUR_SSH_USER@YOUR_WORKSTATION
```

保持此终端打开。`-R` 让工作站的 `127.0.0.1:7899` 转发到自己电脑的代理；代理没有公开监听到工作站外网。如果提示工作站端口被占用，改用 `-R 127.0.0.1:17899:127.0.0.1:7899`，并把下一节工作站侧的代理端口改为 17899。

如果自己电脑实际代理端口不是 7899，只替换 `-R` 的最后一个端口。先确认代理软件支持 HTTP/mixed 还是 SOCKS，按下一节选择一种方式。

## 6. 首次推送与认证

在**工作站发布目录**执行一次推送。HTTP/mixed 代理：

```bash
cd /home/user/cjx/GridPathPlanning-github
git -c http.proxy=http://127.0.0.1:7899 push -u origin main
```

如果代理只支持 SOCKS5，使用：

```bash
git -c http.proxy=socks5h://127.0.0.1:7899 push -u origin main
```

这两条是替代关系，成功一条即可。`-c` 仅对当前 Git 命令生效，不设置全局代理。GitHub HTTPS 认证时，Username 填 GitHub 用户名，Password 输入访问令牌，而非账号密码。可以在 GitHub Settings → Developer settings → Personal access tokens 创建限定本仓库的 fine-grained token，授予 Contents 的 Read and write 权限，并设到期时间；令牌仅在密码提示中输入。

完成后在浏览器确认 README、源码、数据和结果已出现，然后在自己电脑的 SSH 隧道终端按 Ctrl+C 关闭隧道。没有必要额外克隆整个新仓库进行验证，也不必再下载权重。

## 7. 控制流量与后续更新

- pip、预训练模型及上游代码均从工作站自身网络获取；HF/GitHub 源码分别走 README 中的镜像，不借这个上传隧道下载。
- 不使用全局 `ALL_PROXY`、全局 Git 代理或 Git LFS 上传权重。只有 GitHub push 命令指定本地代理。
- 首次提交全部在打开隧道前准备好。后续提交 Git 通常只传新增对象，无需反复删仓库或整包重传。
- 发布包是独立快照，原工作目录后续变化不会自动同步。先筛选并核对新的修改，再提交到发布目录。
- 微调权重另行分享；当前没有公开模型下载链接。README 提供固定预训练版本及从头复现方式，不虚构权重地址。

## 8. 当前交付与验收

入口为 [README](../README.md)。`scripts/package_repository.py` 负责白名单导出与体积统计；`scripts/create_reproduction.py` 创建干净复现目录；`scripts/fetch_pinned_models.py` 在不改写历史模型清单的前提下镜像下载。

发布包包含历史运行元数据，所以从头训练应按 README 先创建独立复现目录，而不是清除原工作站的账本和完成标记。正式训练和长时间评价都由使用者自行执行，本次准备不启动它们。

本地发布目录为 `/home/user/cjx/GridPathPlanning-github`，逐文件体积与哈希为同级的 `GridPathPlanning-github.selection.json`。该目录尚未初始化 Git；按照第 4 节建仓提交即可。尚未新建远程仓库或实际推送。
