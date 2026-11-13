#!/usr/bin/env python3
"""Part 1基线v2完整入口；不访问改进方法的配置、检查点、结果或缓存。"""
import os,sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    from gridpath.experiments.cli import main
    main('v2')
