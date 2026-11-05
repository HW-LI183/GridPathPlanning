#!/usr/bin/env python3
"""创建不含历史运行状态的独立复现目录，不训练、不下载、不修改原目录。"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def create(destination, reuse_assets=False, root=ROOT):
    destination = destination.resolve()
    if destination.exists() or destination == root or root in destination.parents:
        raise ValueError('目标必须是项目之外尚不存在的目录')
    destination.mkdir(parents=True)
    for folder in ('src', 'scripts', 'tests', 'docs', 'configs'):
        shutil.copytree(root / folder, destination / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'primary_model_v1.json', 'primary_model_v1_four.json'))
    for name in ('README.md', 'plan.md', '.gitignore', 'pyproject.toml',
                 'requirements.txt', 'requirements-lock.txt', 'data/v1_sources.json',
                 'manifests/models.json', 'manifests/flan_models.json', 'manifests/upstream_probe.json'):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    if reuse_assets:
        for name in ('models', 'third_party'):
            source = root / name
            if not source.is_dir():
                raise ValueError(f'没有可复用的资源目录：{source}')
            (destination / name).symlink_to(source.resolve(), target_is_directory=True)
    print(f'复现目录：{destination}；未复制检查点、结果、选模锁、训练 lineage 或 GPU 账本。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reuse-assets', action='store_true', help='软链接原目录的预训练模型及上游代码')
    args = parser.parse_args()
    create(args.output, args.reuse_assets)
