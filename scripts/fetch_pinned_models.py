#!/usr/bin/env python3
"""按已发布清单下载固定模型，保留清单原文，避免下载时间等字段改变实验绑定。"""
import copy
import json
from prepare_models import MANIFEST, download


def main():
    original = MANIFEST.read_bytes()
    manifest = copy.deepcopy(json.loads(original))
    for entry in manifest['models']:
        for artifact in entry['files']:
            download(manifest['endpoint'].rstrip('/'), entry, artifact)
    if MANIFEST.read_bytes() != original:
        raise RuntimeError('下载期间模型清单被其他进程修改，请检查')
    print('固定模型全部下载并校验完成；原始清单未改写。')


if __name__ == '__main__':
    main()
