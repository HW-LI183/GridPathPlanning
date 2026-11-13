#!/usr/bin/env python3
"""从公开副本创建空运行目录；保留发布结果，只复制复现输入，不训练或下载。"""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def create(destination,root=ROOT):
    destination=Path(destination).resolve();root=Path(root).resolve()
    if destination.exists() or destination==root or root in destination.parents:
        raise ValueError('目标必须在仓库外且尚不存在，避免覆盖已发布结果')
    destination.mkdir(parents=True)
    for name in ('src','scripts','tests','docs','configs','data','publication'):
        if (root/name).exists():
            shutil.copytree(root/name,destination/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','primary_model*.json'))
    for name in ('README.md','plan.md','.gitignore','pyproject.toml','requirements.txt','requirements-lock.txt'):
        if (root/name).exists():shutil.copy2(root/name,destination/name)
    for name in ('models.json','flan_models.json','upstream_probe.json'):
        source=root/'manifests'/name
        if source.exists():
            target=destination/'manifests'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    # 这是读者自己的独立复现，不继承作者的检查点、选模锁、GPU历史、数据运行清单或结果。
    ledger=destination/'manifests/gpu_budget_total.json';ledger.parent.mkdir(parents=True,exist_ok=True)
    ledger.write_text(json.dumps({'schema_version':'gridpath.gpu_budget.v1','entries':[]},indent=2)+'\n')
    print('独立复现目录已创建；未复制权重、结果、锁或历史GPU记录。')
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--output',type=Path)
    g.add_argument('--carry-v1-budget',action='store_true',help='首次进入v2前继承本独立目录的v1成本，不重置非空账本')
    a=p.parse_args()
    if a.output:create(a.output)
    else:
        source=ROOT/'manifests/gpu_budget.json';target=ROOT/'manifests/gpu_budget_total.json'
        incoming=json.loads(source.read_text());current=json.loads(target.read_text())
        if current.get('entries'):raise ValueError('v2账本非空，拒绝覆盖')
        if any(e['status']=='running' for e in incoming['entries']):raise ValueError('v1会话仍在运行')
        target.write_text(json.dumps(incoming,ensure_ascii=False,indent=2)+'\n')
        print('v1成本已继承到v2/改进账本；未删除历史记录。')
