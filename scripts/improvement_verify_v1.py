#!/usr/bin/env python3
"""Part 2验证改进；复用本轮v2的G，独立训练结构化V，评价后单向比较基线。"""
import os,sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    from gridpath.experiments.cli import main
    main('improvement_verify_v1')
