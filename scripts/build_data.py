#!/usr/bin/env python3
"""阶段三v1短CPU构造，60秒上限。"""
import json,signal,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from gridpath.data import build_dataset
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("短CPU构造超时，请由用户执行")))
signal.alarm(60)
try:
 result=build_dataset()
 print(json.dumps({"version":result["version"],"counts":{k:v["count"] for k,v in result["files"].items()},"cpu_seconds":result["build_cpu_seconds"]},ensure_ascii=False,indent=2))
finally:signal.alarm(0)
