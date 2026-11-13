"""CPU准备：构造/审计数据、检查本地分词与磁盘；不训练、不运行模型前向。"""
import shutil
from ..training_config import read_json,write_json,model_keys
from ..data_schema import grid_from_record
from ..planning.protocol import feature
from .data import ROOT,audit,load
from .config import config_for,pretrained_card,completed_model,LEDGER


def prepare(method,selected=None):
    from transformers import AutoTokenizer
    checked=audit();d=load();models={}
    for model in (selected or model_keys()):
        cfg=config_for(method,model)
        pretrained_card(model,cfg,True)
        if method!='v2':completed_model('v2',model,'generate')
        tok=AutoTokenizer.from_pretrained(ROOT/cfg['model_directory'],local_files_only=True)
        plain_inputs=plain_targets=structured=0
        for row in d['train_generate']+d['train_verify']+d['dev_generate']+d['dev_verify']+d['dev_repair_seeds']:
            if method!='v2' and row['task']!='GENERATE':
                f=feature(tok,grid_from_record(row),row['candidate_plan'],cfg['verify_source_tokens'])
                structured=max(structured,len(f['input_ids']))
            if method=='v2':
                n=len(tok(row['input_text'],truncation=False)['input_ids']);plain_inputs=max(plain_inputs,n)
                target=len(tok(row['target_text'],truncation=False)['input_ids']);plain_targets=max(plain_targets,target)
                if n>cfg['max_source_tokens'] or target>(256 if row['task']=='GENERATE' else 64):raise ValueError('输入/目标超长，不截断')
        # 封存测试只检查固定输入可序列化及长度，不运行模型，不按结果调整任何配置。
        for split in ('test_id','test_ood'):
            for row in d[split+'_verify']+d[split+'_repair_seeds']:
                if method!='v2':structured=max(structured,len(feature(tok,grid_from_record(row),row['candidate_plan'],cfg['verify_source_tokens'])['input_ids']))
                elif len(tok(row['input_text'],truncation=False)['input_ids'])>cfg['max_source_tokens']:raise ValueError('固定测试输入超长')
        models[model]={'max_plain_tokens':plain_inputs,'max_target_tokens':plain_targets,'max_structured_tokens':structured,
                       'new_G_training':method=='v2','V_updates':750,'G_updates':2250 if method=='v2' else 0}
    entries=read_json(LEDGER)['entries'];hours=sum(e.get('wall_seconds',0) for e in entries)/3600
    if any(e['status']=='running' for e in entries) or hours>=20:raise ValueError('GPU会话未结账或累计预算已用尽')
    # baseline: G+V各四个完整seq2seq；改进: 四个编码器头，latest+原子写临时空间。
    params=[222903552,60506624,247577856,76961152] if method=='v2' else [109633158,35333894,109633158,35335878]
    params=[dict(zip(model_keys(),params))[m] for m in (selected or model_keys())]
    required=sum(params)*12*(2 if method=='v2' else 1)+max(params)*12+1_000_000_000
    existing=sum(p.stat().st_size for p in (ROOT/f'checkpoints/{method}').rglob('*') if p.is_file() and not p.is_symlink())
    free=shutil.disk_usage(ROOT).free
    if free+existing<required:raise ValueError(f'可用空间不足，需约{required/1e9:.2f}GB（含原子快照）；归档仍占同一磁盘空间')
    result={'method':method,'status':'passed','models':models,'data':checked,'gpu_calls':0,
       'gpu_hours_used':hours,'gpu_hours_remaining':20-hours,'free_bytes':free,'storage_reserve_bytes':required,
       'dependency':'仅原始预训练资源与共同冻结数据' if method=='v2' else '本轮v2完成的G检查点；V从原始预训练编码器开始'}
    write_json(ROOT/f'manifests/preflight_{method}.json',result);return result
