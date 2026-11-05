#!/usr/bin/env python3
"""按白名单生成独立 GitHub 上传目录，离线统计体积；不创建仓库或上传。"""
import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def selection(root=ROOT):
    selected = set()
    for name in ('README.md', 'plan.md', '.gitignore', 'pyproject.toml',
                 'requirements.txt', 'requirements-lock.txt'):
        selected.add(root / name)
    for folder in ('configs', 'docs', 'src', 'scripts', 'tests', 'data', 'results/v1'):
        selected.update(p for p in (root / folder).rglob('*') if p.is_file())
    selected.update(p for p in (root / 'manifests').glob('*.json') if p.is_file())
    # 保留轻量训练证据；不遍历 _states，也不递归跟随 best/latest 链接。
    for model in ('t5_base', 't5_small'):
        base = root / 'checkpoints/v1' / model
        selected.update(base.glob('*.json'))
        selected.update(base.glob('*.jsonl'))
        selected.update((base / 'evaluations').glob('*.json'))
        if (base / 'best/checkpoint.json').is_file():
            selected.add(base / 'best/checkpoint.json')
    return sorted(p for p in selected if p.is_file()
                  and '__pycache__' not in p.parts
                  and p.suffix not in ('.pyc', '.part', '.lock')
                  and not any(part.startswith('.env') for part in p.relative_to(root).parts))


def export(destination, root=ROOT):
    destination = destination.resolve()
    if destination.exists() or destination == root or root in destination.parents:
        raise ValueError('请使用项目目录之外、尚不存在的目标目录，避免覆盖已有工作')
    files = selection(root)
    oversized = [str(p) for p in files if p.stat().st_size >= 50_000_000]
    if oversized:
        raise ValueError(f'白名单出现意外大文件，停止：{oversized}')
    destination.mkdir(parents=True)
    records = []
    for source in files:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({'path': relative.as_posix(), 'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    # 仅在内存估算 gzip 后体积；不是 Git 实际传输量，也不上传压缩包。
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as archive:
            for record in records:
                archive.add(destination / record['path'], arcname=record['path'], recursive=False)
    report = {'destination': str(destination), 'file_count': len(records),
              'total_bytes': sum(item['bytes'] for item in records),
              'tar_gzip_bytes_estimate': len(buffer.getvalue()), 'files': records}
    report_path = destination.with_name(destination.name + '.selection.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report, report_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, report_file = export(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False, indent=2))
    print(f'逐文件大小及 SHA-256 清单：{report_file}')
