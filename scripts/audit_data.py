#!/usr/bin/env python3
"""阶段三v1短CPU独立审计，60秒上限。"""
import json,signal,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from gridpath.data_audit import audit_dataset
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("短CPU审计超时，请由用户执行")))
signal.alarm(60)
try:
 result=audit_dataset()
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if result["status"]!="passed":raise SystemExit(1)
finally:signal.alarm(0)
