"""仅兼容已冻结的v1源码身份；当前方法不使用旧路径重定向。"""
from functools import lru_cache
from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[3]
RECEIPT = PROJECT / 'manifests/baselines/v1_runtime_compatibility.json'


@lru_cache(maxsize=2)
def _read_layout(mtime, size):
    return json.loads(RECEIPT.read_text())


def layout():
    if not RECEIPT.exists():return {'path_mappings':{},'source_files':{}}
    stat=RECEIPT.stat()
    return _read_layout(stat.st_mtime_ns,stat.st_size)


def _translate(key, pairs):
    for old,new in sorted(pairs,key=lambda x:len(x[0]),reverse=True):
        if key==old or key.startswith(old+'/'):return new+key[len(old):]
    return key


def resolve_artifact(path):
    path=Path(path)
    try:key=path.relative_to(PROJECT).as_posix()
    except ValueError:return path
    return PROJECT / _translate(key,layout()['path_mappings'].items())


def identity_path(path):
    path=Path(path)
    key=path.relative_to(PROJECT).as_posix()
    return _translate(key,[(new,old) for old,new in layout()['path_mappings'].items()])


class WorkspacePath(type(Path())):
    def __truediv__(self,key):
        try:base=PROJECT/identity_path(self)
        except ValueError:base=Path(self)
        return WorkspacePath(resolve_artifact(base/key))


def original_source_sha(path):
    """校验本次纯迁移的实际文件后，返回迁移前的运行源码哈希。"""
    import hashlib
    physical=resolve_artifact(path)
    current=hashlib.sha256(physical.read_bytes()).hexdigest() if physical.exists() else None
    try:key=identity_path(physical)
    except ValueError:return current
    witness=layout()['source_files'].get(key)
    if witness:
        if current!=witness.get('after_sha256'):raise ValueError('迁移后的源码未获登记：'+key)
        return witness['before_sha256']
    return current
