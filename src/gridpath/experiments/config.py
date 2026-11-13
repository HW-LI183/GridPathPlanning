"""方法配置、实际源码指纹与三遍预算；基线不访问改进配置或产物。"""
from pathlib import Path
import importlib.metadata
import yaml
from ..training_config import read_json,write_json,sha256,digest,model_keys
from .data import ROOT,MANIFEST,load

METHODS=('v2','improvement_verify_v1')
LEDGER=ROOT/'manifests/gpu_budget_total.json'


def config_for(method,model,role='verify'):
    if method not in METHODS or model not in model_keys() or role not in ('generate','verify'):raise ValueError('未知方法、模型或角色')
    if method!='v2' and role=='generate':raise ValueError('改进只训练V；G来自本轮v2')
    p=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text())
    if (p['seed']!=42 or p['max_total_epochs']!=3 or p['effective_batch_size']!=4 or p['micro_batch_size']!=4 or
        p['train_generate']!=3000 or p['train_verify']!=1000 or p['gpu_hours_cap']!=20 or
        p['split_panels']!={'generate':60,'verify':80,'repair':60}):raise ValueError('固定4000训练记录、各200评价记录、三遍及累计20 GPU-hours')
    recipe=yaml.safe_load((ROOT/f'configs/{method}/training.yaml').read_text())
    if recipe['method']!=method:raise ValueError('方法配置身份不符')
    card=yaml.safe_load((ROOT/'configs/baselines/models.yaml').read_text())['models'][model]
    n=p['train_'+role];target=recipe['path_target_tokens'] if role=='generate' else recipe['verification_target_tokens']
    return {**p,**recipe,**card,'method':method,'model_key':model,'role':role,
        'run_directory':f'checkpoints/{method}/{model}/{role}','max_updates':n*3//4,
        'evaluate_steps':[n//4,2*n//4,3*n//4], 'max_target_tokens':target,'generation_tokens':target,
        'structured':method=='improvement_verify_v1'}


def pretrained_card(model,cfg,verify_weights=False):
    cards=read_json(ROOT/'manifests'/('flan_models.json' if model.startswith('flan_') else 'models.json'))['models']
    card=next(c for c in cards if c['repo_id']==cfg['model_id'])
    if card['revision']!=cfg['revision'] or card['local_directory']!=cfg['model_directory']:raise ValueError('预训练来源不符')
    for f in card['files']:
        path=ROOT/card['local_directory']/f['name']
        if not path.is_file():raise FileNotFoundError(f'缺少预训练资源；先运行 bash scripts/run_phase4_v1.sh prepare {model}')
        if f['name']=='model.safetensors' and not verify_weights:continue
        if sha256(path)!=f['sha256']:raise ValueError('预训练文件损坏')
    return card


def source_files(method):
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/f'configs/{method}/training.yaml',ROOT/'configs/experiment.yaml',
        ROOT/'configs/baselines/models.yaml',ROOT/'src/gridpath/v2/prompts.py',ROOT/'src/gridpath/planning/events.py',
        ROOT/'src/gridpath/planning/protocol.py',ROOT/'src/gridpath/baselines/feedback.py',ROOT/'src/gridpath/baselines/layout.py']
    files += [ROOT/f'src/gridpath/{n}.py' for n in ('training_helpers','training_config','inference','evaluation','metrics',
        'executor','types','data_schema','bfs','prompts','corruptions','resources')]
    if method!='v2':files += [ROOT/'src/gridpath/planning/model.py',ROOT/'src/gridpath/planning/verification.py']
    script='phase4_v2' if method=='v2' else 'improvement_verify_v1'
    files += [ROOT/f'scripts/{script}.py',ROOT/f'scripts/run_{script}.sh',ROOT/'scripts/run_experiment.sh']
    return sorted(set(files))


def binding_for(method,model,role='verify',verify_weights=False):
    cfg=config_for(method,model,role);dev_key='dev_generate' if role=='generate' else 'dev_verify'
    d=load(['train_'+role,dev_key]);train=d['train_'+role];dev=d[dev_key]
    payload={'config':cfg,'dataset':sha256(MANIFEST),'model':pretrained_card(model,cfg,verify_weights),
        'sources':{str(p.relative_to(ROOT)):sha256(p) for p in source_files(method)},
        'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy')}}
    if method!='v2':
        _,g,b=completed_model('v2',model,'generate')
        payload['baseline_G']={'training_fingerprint':b['fingerprint'],'files':read_json(g/'checkpoint.json')['files']}
    return cfg,train,dev,{'fingerprint':digest(payload),**payload}


def completed_model(method,model,role='verify'):
    cfg,_,_,binding=binding_for(method,model,role)
    run=ROOT/cfg['run_directory']
    if not (run/'completed.json').exists():raise FileNotFoundError(f'尚未完成 {method}/{model}/{role}；请按文档从头训练')
    done=read_json(run/'completed.json');meta=read_json(run/'latest/checkpoint.json')
    n=cfg['train_'+role]
    if (done['fingerprint']!=binding['fingerprint'] or meta['binding']!=binding or done['state']!=meta['state'] or
        done['state']['global_step']!=cfg['max_updates'] or done['state']['consumed_examples']!=3*n):
        raise ValueError('必须使用相同来源的完整第三遍末权重')
    for name,expected in meta['files'].items():
        if name=='training_state.pt':continue
        if sha256(run/'latest'/name)!=expected:raise ValueError('末遍推理文件损坏')
    return cfg,run/'latest',binding


def budget(method,model,stage):
    from ..resources import GPUBudget
    cfg=config_for(method,model)
    return GPUBudget(LEDGER,cfg['gpu_hours_cap'],cfg['session_wall_seconds'],f'{method}:{model}:{stage}')
