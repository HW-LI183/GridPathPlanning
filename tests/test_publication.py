"""发布工具只做本地隔离复制，防止权重进入上传包或历史状态污染复现。"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_selection_excludes_heavy_assets(tmp_path):
    for name in ('data/v1_sources.json', 'results/v1/summary_dev.json',
                 'checkpoints/v1/t5_base/best/checkpoint.json',
                 'models/model.safetensors', 'checkpoints/v1/t5_base/_states/x/model.safetensors',
                 'logs/debug.log', 'src/gridpath/__pycache__/x.pyc'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{}')
    selected = load('package_repository').selection(tmp_path)
    relative = [p.relative_to(tmp_path).as_posix() for p in selected]
    assert 'data/v1_sources.json' in relative
    assert 'results/v1/summary_dev.json' in relative
    assert 'checkpoints/v1/t5_base/best/checkpoint.json' in relative
    assert not any(p.endswith(('.pt', '.safetensors', '.pyc', '.pdf')) and not p.startswith('results/v1/figures/') for p in relative)
    assert not any(p.startswith(('models/', 'third_party/', 'logs/')) for p in relative)


def test_reproduction_isolates_history(tmp_path):
    target = tmp_path / 'fresh'
    load('create_reproduction').create(target)
    for name in ('configs/primary_model_v1.json', 'manifests/v1_training_lineage.json',
                 'manifests/gpu_budget.json', 'results', 'checkpoints'):
        assert not (target / name).exists()
    assert (target / 'data/v1_sources.json').read_bytes() == (ROOT / 'data/v1_sources.json').read_bytes()
    assert (target / 'manifests/models.json').read_bytes() == (ROOT / 'manifests/models.json').read_bytes()


def test_reproduction_refuses_existing_destination(tmp_path):
    with pytest.raises(ValueError):
        load('create_reproduction').create(tmp_path)


def test_export_refuses_existing_destination(tmp_path):
    with pytest.raises(ValueError):
        load('package_repository').export(tmp_path)
