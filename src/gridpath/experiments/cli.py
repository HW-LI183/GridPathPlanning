"""两套方法统一接口；基线没有比较入口，也不读取改进的模型、结果或缓存。"""
import argparse,json
from ..training_config import model_keys
from .data import SPLITS,ROOT


def main(method):
    modes=['prepare','pilot','train','evaluate','test','summarize','select','review']
    if method!='v2':modes.append('compare')
    p=argparse.ArgumentParser(description='从原始预训练资源运行 '+method+'；正式任务由用户启动，无性能门槛')
    p.add_argument('mode',choices=modes);p.add_argument('--model',choices=model_keys(),default='t5_base')
    p.add_argument('--models',nargs='+',choices=model_keys(),help='准备、汇总、选择、复核、比较的模型集合；默认全部')
    p.add_argument('--role',choices=['generate','verify'] if method=='v2' else ['verify'],default='verify')
    p.add_argument('--split',choices=SPLITS,default='dev');a=p.parse_args()
    if a.models and (len(set(a.models))!=len(a.models) or a.mode in ('train','pilot','evaluate','test')):p.error('--models用于集合操作；单模型推理/训练请用--model')
    if a.mode in ('train','pilot') and a.split!='dev':p.error('训练只读取train，开发诊断固定dev；不能用测试分区训练')
    if a.mode=='prepare':
        from .preparation import prepare
        result=prepare(method,a.models)
    elif a.mode in ('pilot','train'):
        from .training import run
        result=run(method,a.model,a.role,a.mode=='pilot')
    else:
        from . import evaluation as e
        if a.mode=='evaluate' and a.split!='dev':p.error('测试分区请使用test命令，先冻结开发选模')
        if a.mode=='test' and a.split=='dev':p.error('test要求--split test_id或test_ood')
        if a.mode in ('evaluate','test'):
            r=e.run(method,a.model,a.split);result={k:r[k] for k in ('method','model','split','record_budget')}
        elif a.mode=='summarize':result={k:v for k,v in e.summarize_files(method,a.split,a.models).items() if k!='models'}
        elif a.mode=='select':
            if a.split!='dev':p.error('只允许开发集选模')
            result=e.select_primary(method,models=a.models)
        elif a.mode=='review':result=e.review(method,a.split,a.models)
        else:result=e.compare(a.split,a.models)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result.get('status')=='stopped':raise SystemExit(130)
