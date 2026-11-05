#!/usr/bin/env python3
"""按已发布清单下载固定模型，保留清单原文，避免下载时间等字段改变实验绑定。"""
import argparse
import copy
import json
from prepare_models import MANIFEST, download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--manifest", help="按指定文件清单下载")
    group.add_argument("--model", choices=("t5_base","t5_small","flan_t5_base","flan_t5_small"), help="下载一个基线的固定资源")
    args = parser.parse_args()
    from pathlib import Path
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST
    if args.model and args.model.startswith("flan_"):
        manifest_path = MANIFEST.with_name("flan_models.json")
    original = manifest_path.read_bytes()
    manifest = copy.deepcopy(json.loads(original))
    for entry in manifest['models']:
        if args.model and entry['local_directory'].rsplit('/',1)[-1].replace('-','_') != args.model:
            continue
        for artifact in entry['files']:
            download(manifest['endpoint'].rstrip('/'), entry, artifact)
    if manifest_path.read_bytes() != original:
        raise RuntimeError('下载期间模型清单被其他进程修改，请检查')
    print('固定模型全部下载并校验完成；原始清单未改写。')


if __name__ == '__main__':
    main()
