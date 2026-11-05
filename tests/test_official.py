"""官方函数加载只能执行审阅过的函数定义，拒绝源码漂移。"""

import hashlib

import pytest

from gridpath.official import load_official_is_goal


def test_loader_does_not_import_or_run_the_upstream_module(tmp_path):
    source = tmp_path / "upstream.py"
    source.write_text("import a_package_that_does_not_exist\nraise RuntimeError('顶层不应执行')\n"
                      "def is_goal(*args):\n    return 1\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert load_official_is_goal(source, expected_sha256=digest)([]) == 1


def test_loader_rejects_changed_source_before_any_execution(tmp_path):
    source = tmp_path / "upstream.py"
    source.write_text("raise RuntimeError('不应执行')\n")
    with pytest.raises(ValueError, match="哈希"):
        load_official_is_goal(source, expected_sha256="0" * 64)
