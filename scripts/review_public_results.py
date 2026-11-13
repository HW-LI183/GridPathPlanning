#!/usr/bin/env python3
"""只用已发布预测和执行器重算指标；不需要权重、不前向、不修改结果。"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def review(root=ROOT):
    from gridpath.training_config import read_json,read_rows,config_for
    from gridpath.evaluation import summarize_panel
    from gridpath.experiments.evaluation import evaluate_with_predictors
    from gridpath.experiments import evaluation as experiment_evaluation
    from gridpath.experiments.ledger import RecordLedger
    from unittest.mock import patch
    from gridpath.experiments.config import config_for as experiment_config
    from gridpath.experiments.data import load
    report={'gpu_calls':0,'checked':[]}
    v1=read_json(root/'results/v1/summary_dev.json')
    data=read_rows(root/'data/v1/dev.jsonl')
    for model,stored in v1['models'].items():
        d=root/f'results/v1/{model}/dev'
        cases=[read_json(d/'cases'/(r['sample_id']+'.json')) for r in data]
        for key,task in [('generated','GENERATE'),('controlled_repair','REPAIR')]:
            rows=[r for r in data if r['task']==task]
            computed=summarize_panel(rows,cases,config_for(model))
            if computed!=stored['panels'][key]:raise ValueError('v1汇总重算不符：'+model+'/'+key)
        report['checked'].append('v1/'+model+'/dev')
    def no_model(*args):raise AssertionError('离线复核缺少已发布IO，禁止调用模型')
    for method in ('v2','improvement_verify_v1'):
        for split in ('dev','test_id','test_ood'):
            total=read_json(root/f'results/{method}/summary_{split}.json')
            dataset=load([split+'_'+k for k in ('generate','verify','repair_seeds','records','samples')])
            for model,stored in total['models'].items():
                d=root/f'results/{method}/{model}/{split}';binding=read_json(d/'binding.json')
                cfg={**experiment_config('v2',model,'generate'),'method':method}
                class FrozenLedger(RecordLedger):
                    def put(self,key,value):
                        expected={'slot':self.slots[key],'fingerprint':self.binding,**value}
                        old=self.get(key)
                        if old!=expected:raise ValueError('已发布调用记录缺失或不一致')
                        return old
                with patch.object(experiment_evaluation,'RecordLedger',FrozenLedger):
                    computed=evaluate_with_predictors(model,cfg,dataset,split,no_model,no_model,d,binding,False)
                if computed!={k:v for k,v in stored.items() if k!='performance'}:raise ValueError('指标重算不符：'+method+'/'+model+'/'+split)
                report['checked'].append(method+'/'+model+'/'+split)
    report['status']='passed'
    return report


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(review(),ensure_ascii=False,indent=2))
