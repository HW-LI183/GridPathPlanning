#!/usr/bin/env python3
"""CPU核验并扩展开发比较集合；保留已测试模型、冻结配置及原选择历史。"""
import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gridpath.training_config import read_json,write_json,sha256,digest,model_keys
from gridpath.experiments import evaluation as evaluation
from gridpath.experiments.data import ROOT,MANIFEST


def expand(method,models,apply=False):
    models=tuple(m for m in model_keys() if m in models)
    lock=ROOT/f'configs/{method}/primary_model.json'
    marker=ROOT/f'manifests/test_usage_{method}.json'
    previous=read_json(lock)
    proposed=evaluation.select_primary(method,False,models)
    old={c['model']:c for c in previous['candidates']}
    new={c['model']:c for c in proposed['candidates']}
    if not old.keys()<=new.keys():raise ValueError('仅允许增加模型，不能删除已冻结模型')
    if any(old[m]!=new[m] for m in old):raise ValueError('已冻结模型的开发分数发生变化，不能扩展')
    frozen=read_json(marker) if marker.exists() else None
    if frozen and frozen!={'method':method,'dataset':sha256(MANIFEST),'dev_selection':sha256(lock)}:
        raise ValueError('测试数据或冻结记录身份不符，不能重置测试记录')
    total=read_json(ROOT/f'results/{method}/summary_dev.json')
    bindings={}
    for model in models:
        _,_,binding,_,_=evaluation.evaluation_binding(method,model,'dev')
        directory=evaluation.directory_for(method,model,'dev')
        if (read_json(directory/'binding.json')!=binding or
            read_json(directory/'summary.json')!=total['models'][model] or
            total['models'][model]['fingerprint']!=binding['fingerprint']):
            raise ValueError('开发结果与当前数据/检查点不一致：'+model)
        bindings[model]=binding['fingerprint']
        for split in ('test_id','test_ood'):
            directory=evaluation.directory_for(method,model,split)
            if not (directory/'binding.json').exists():continue
            _,_,expected,_,_=evaluation.evaluation_binding(method,model,split)
            if read_json(directory/'binding.json')!=expected:raise ValueError('已有测试输入或检查点变化：'+model+'/'+split)
            if (directory/'summary.json').exists() and read_json(directory/'summary.json')['fingerprint']!=expected['fingerprint']:
                raise ValueError('已有测试成绩身份不符：'+model+'/'+split)
    result={'method':method,'status':'unchanged' if previous==proposed else 'validated',
        'old_selection':previous,'new_selection':proposed,'old_test_usage':frozen,
        'dataset':sha256(MANIFEST),'dev_bindings':bindings,'gpu_calls':0,
        'policy':'仅增加预注册模型；已有模型开发分数、数据、源码和检查点不变；只按dev排序，不读取测试指标选模'}
    if apply and previous!=proposed:
        receipt=ROOT/f'manifests/maintenance/model_set_expansion_{method}_{digest(result)[:16]}.json'
        write_json(receipt,result)
        write_json(lock,proposed)
        if frozen:write_json(marker,{**frozen,'dev_selection':sha256(lock)})
        result.update(status='applied',receipt=str(receipt.relative_to(ROOT)))
        write_json(receipt,result)
    return result


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=('v2','improvement_verify_v1'),required=True)
    p.add_argument('--models',nargs='+',choices=model_keys(),default=list(model_keys()))
    p.add_argument('--apply',action='store_true',help='核验后更新选择与测试身份；默认只检查')
    args=p.parse_args()
    lock=ROOT/'manifests/gpu_budget_total.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=expand(args.method,args.models,args.apply)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
