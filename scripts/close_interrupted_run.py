#!/usr/bin/env python3
"""旧 GPU 进程确认停止后，为未正常结束的会话保守结账。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--confirm-stopped", action="store_true", required=True)
    args = parser.parse_args()
    from gridpath.resources import close_interrupted
    from gridpath.training_config import ROOT
    result = close_interrupted(ROOT / "manifests/gpu_budget.json", args.entry_id)
    print(f"会话已保守结账：{result['wall_seconds']:.1f} 秒；训练暴露记录保持原样")


if __name__ == "__main__":
    main()
