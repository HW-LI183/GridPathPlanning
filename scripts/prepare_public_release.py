#!/usr/bin/env python3
"""构造通用、经过隐私检查的公开副本；不联网、不训练、不修改原实验。"""
import argparse,gzip,hashlib,io,json,re,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOCS=('phase1.md','phase2.md','phase_template.md','phase3.md','phase4.md','phase3_v2.md','phase4_v2.md','improvement_verify_t5base_v1.md','improvement_verify_all_v1.md')
SCRIPTS=('phase4.py','run_phase4_v1.sh','phase4_v2.py','run_phase4_v2.sh','improvement_verify_v1.py',
 'run_improvement_verify_v1.sh','run_experiment.sh','run_model_set_expansion.sh','expand_model_set.py',
 'build_data.py','audit_data.py','fetch_pinned_models.py','prepare_models.py','check_flan_models.py',
 'close_interrupted_run.py','create_public_reproduction.py','review_public_results.py','prepare_public_release.py',
 'check_phase1.py','check_phase2.py','inspect_upstream.py')
TESTS=('test_bfs.py','test_executor.py','test_metrics.py','test_corruptions.py','test_types.py',
 'test_data_schema.py','test_step_verifier.py','test_resources.py','test_official.py')
MANIFESTS=('models.json','flan_models.json','upstream_probe.json','dataset_manifest.json','dataset_v2.json','data_audit_v2.json',
 'phase1_recheck.json','phase1_resources.json','phase2_checks.json')
PRIVATE_PATH=re.compile(r'/(?:home|Users|root|mnt)/[A-Za-z0-9_.-]+(?:/[^\s\"\'`<>]*)?')
SECRET=re.compile(r'(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]{20,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----')
CONDA_PATH=re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/(?:mini|ana)conda[^/\s]*/bin/conda')


def sha(content):return hashlib.sha256(content).hexdigest()


def sanitize(text,root):
    text=CONDA_PATH.sub('conda',text).replace(str(root.resolve()),'.')
    return text.replace('实验设备','实验设备')


def public_document(text,root):
    text=sanitize(text,root)
    if text.startswith('# 阶段 1：'):
        text=re.sub(r'### 2\.1 .*?(?=\n### 2\.2)',
            '### 2.1 复现环境\n\n参考GPU为RTX A6000；Python3.10、PyTorch2.5.1+CUDA12.1，支持BF16；seed42、20 GPU-hours预算。Conda使用PATH中的`conda`或`GRIDPATH_CONDA`，无需作者的设备路径、环境克隆来源或连接信息。\n',text,flags=re.S)
        text=re.sub(r'### 3\.1 .*?(?=\n### 3\.2)',
            '### 3.1 独立Conda环境\n\n按[公开README](../README.md#5-环境与权重)创建干净的`gridpath`环境，安装固定依赖。公开复现不依赖任何已有私人环境或其他项目；本机环境转储不随仓库发布。\n',text,flags=re.S)
        text=re.sub(r'### 3\.5 .*?(?=\n## 4\.)',
            '### 3.5 文件清单\n\n公开配置和依赖见`configs/project.json`、`requirements.txt`和`requirements-lock.txt`；模型来源见`manifests/models.json`与`flan_models.json`。资源状态与文件核验的公开摘要见`phase1_resources.json`、`phase1_recheck.json`；上游语义探查见`upstream_probe.json`。检查入口为`check_phase1.py`、`inspect_upstream.py`。私人环境转储和终端日志不公开。\n',text,flags=re.S)
        text=re.sub(r'## 7\. .*?(?=\n## 8\.)',
            '## 7. 问题与后续处理\n\n模型家族、revision和下载哈希以公开清单为准。镜像下载中断可续传；不得改成未固定版本。上游布局隔离、非法词处理、距离口径及安全加载问题已经记录在第3节，并由阶段2的严格执行器和BFS处理。下载与真实模型加载需要读者自行执行，不是查阅公开结果的前置条件。\n',text,flags=re.S)
    # 执行说明集中维护，移除会覆盖已发表结果的本地恢复命令和环境准备历史。
    text=re.sub(r'## 4\. 执行命令.*?(?=\n## 5\.)',
        '## 4. 执行命令\n\n先创建独立复现目录；环境、数据、训练与评价的完整顺序见[复现说明](reproduce.md)。公开结果只读保留，不在此目录继续训练或覆盖汇总。\n',text,flags=re.S)
    text=re.sub(r'## 6\. 资源使用记录.*?(?=\n## 7\.)',
        '## 6. 资源使用记录\n\n实验参考GPU为RTX A6000；样本暴露、训练更新与成本摘要见`results/training_summary.json`。公开副本不提供设备名称、系统路径、环境转储、空闲磁盘或连接配置。读者独立运行应记录自己的实际成本。\n',text,flags=re.S)
    text=text.replace('dev_overview.pdf','dev_overview.png')
    text=text.replace('manifests/gpu_budget_total.json','results/training_summary.json')
    parts=text.split('\n\n')
    parts=[p for p in parts if not any(k in p for k in ('manifests/maintenance/','rerun_reset.json','manifests/v1_training_lineage.json','source_snapshots/','v1_runtime_compatibility.json'))]
    return '\n\n'.join(parts).rstrip()+'\n'


def public_check_manifest(name,text):
    """公开准备阶段的科学检查摘要，不发布个人进程路径或原始环境转储。"""
    data=json.loads(text)
    if name=='phase1_recheck.json':
        keys=('phase','status','mode','runtime_phase1_acceptance','files','models','verified_file_count',
              'verified_weight_count','verified_total_bytes','remaining_pending_weight_bytes')
        data={k:data[k] for k in keys if k in data}
    elif name=='phase1_resources.json':
        keys=('phase','status','training_runs','gpu_pilot_runs','batch_evaluation_runs','gpu_environment_checks',
              'budget_charge_gpu_hours','budget_charge_note','downloaded_model_bytes','pending_weight_bytes')
        data={k:data[k] for k in keys if k in data}
    else:data.pop('python_executable',None)
    data['public_note']='原阶段检查记录的公开摘要；已排除个人运行路径和环境转储，不代表本次重新加载模型。'
    return json.dumps(data,ensure_ascii=False,indent=2)+'\n'


def selection(root=ROOT):
    root=Path(root);files=set()
    for name in ('pyproject.toml','requirements.txt','requirements-lock.txt'):
        files.add(root/name)
    for name in ('src','configs','data'):
        files.update(p for p in (root/name).rglob('*') if p.is_file())
    files.update(root/'scripts'/n for n in SCRIPTS)
    files.update(root/'tests'/n for n in TESTS)
    files.update(root/'docs'/n for n in DOCS)
    files.update(root/'manifests'/n for n in MANIFESTS)
    files.update(p for p in (root/'results').rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.png','.svg'))
    files.update(p for p in (root/'publication').glob('*.md'))
    excluded={'cache','generate_cache','verify_cache','__pycache__','_states'}
    return sorted(p for p in files if p.is_file() and not (excluded & set(p.relative_to(root).parts))
      and p.suffix in ('.py','.sh','.json','.jsonl','.yaml','.yml','.md','.toml','.txt','.csv','.png','.svg')
      and not any(part.startswith('.env') or part=='.ssh' for part in p.relative_to(root).parts)
      and not p.name.startswith('primary_model')
      and p.relative_to(root).as_posix() not in ('results/workspace_audit.json','results/v1/preflight.json'))


def audit_release(root):
    root=Path(root);findings=[]
    for p in root.rglob('*'):
        if {'__pycache__','.pytest_cache'} & set(p.relative_to(root).parts):continue
        if p.is_symlink():findings.append(str(p.relative_to(root))+': symlink');continue
        if not p.is_file():continue
        rel=p.relative_to(root)
        if rel.parts[0] in ('models','checkpoints','logs','third_party') or p.suffix in ('.pt','.pth','.bin','.safetensors','.pem','.key'):
            findings.append(str(rel)+': excluded artifact')
        text=p.read_bytes().decode('utf-8',errors='ignore')
        if PRIVATE_PATH.search(text) or SECRET.search(text):findings.append(str(rel)+': private path or credential pattern')
    if findings:raise ValueError('公开检查未通过：'+json.dumps(findings[:30],ensure_ascii=False))
    return {'status':'passed','checked_files':sum(p.is_file() for p in root.rglob('*')),'private_path_or_secret_matches':0,'symlinks':0}


def export(destination,root=ROOT):
    root=Path(root).resolve();destination=Path(destination).resolve()
    if destination.exists() or destination==root or root in destination.parents:
        raise ValueError('发布目标必须在项目外且尚不存在，不能覆盖实验或已有发布目录')
    files=selection(root)
    if any(p.stat().st_size>=50_000_000 for p in files):raise ValueError('发现超过本工具50MB单文件上限的内容，请单独审阅')
    destination.mkdir(parents=True);changes=[]
    for source in files:
        rel=source.relative_to(root);target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
        raw=source.read_bytes();new=raw
        if source.suffix in ('.md','.py','.sh','.yaml','.json','.toml','.txt'):
            original=raw.decode('utf-8')
            content=public_document(original,root) if rel.parts[0]=='docs' else sanitize(original,root)
            if rel.parts[0]=='manifests' and source.name in ('phase1_recheck.json','phase1_resources.json','phase2_checks.json'):
                content=public_check_manifest(source.name,content)
            new=content.encode('utf-8')
        target.write_bytes(new);shutil.copymode(source,target)
        if new!=raw:changes.append({'path':str(rel),'original_sha256':sha(raw),'public_sha256':sha(new),'reason':'generic paths or public documentation'})
    for name,targetname in [('README.md','README.md'),('reproduce.md','docs/reproduce.md'),('plan.md','plan.md')]:
        target=destination/targetname;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text((root/'publication'/name).read_text())
    # 公开副本不使用原实验运行身份，保存的是预测证据与轻量训练计数。
    training={}
    for marker in (root/'checkpoints').rglob('completed.json'):
        if '_states' in marker.parts:continue
        row=json.loads(marker.read_text());name=str(marker.parent.relative_to(root/'checkpoints'))
        training[name]={k:row[k] for k in ('status','state','seconds') if k in row}
    ledger=json.loads((root/'manifests/gpu_budget_total.json').read_text())
    training_report={'training_runs':training,'recorded_project_gpu_hours':sum(e['wall_seconds'] for e in ledger['entries'])/3600,
        'cost_note':'原实验实际累计；不包含个人路径或连接配置。读者独立运行另记成本。'}
    training_report['current_method_stage_seconds']={method:{role:sum(e['wall_seconds'] for e in ledger['entries']
        if e['label'].startswith(method+':') and len(e['label'].split(':'))>=3 and e['label'].split(':')[2]==role)
        for role in ('generate','verify','dev','test_id','test_ood')} for method in ('v2','improvement_verify_v1')}
    (destination/'results/training_summary.json').write_text(json.dumps(training_report,ensure_ascii=False,indent=2)+'\n')
    (destination/'.gitignore').write_text('models/\ncheckpoints/\nlogs/\nthird_party/\n__pycache__/\n.pytest_cache/\n*.py[cod]\n*.pt\n*.pth\n*.bin\n*.safetensors\n.env*\n*.pem\n*.key\n*.selection.json\nresults/**/cache/\nresults/**/generate_cache/\nresults/**/verify_cache/\n')
    provenance={'schema':'gridpath.public_release.v1','weights_included':False,'checkpoints_included':False,
        'results_policy':'原始预测、CSV和JSON数值原样保留；只排除本机预检、工作区状态与推理缓存',
        'fingerprint_policy':'原始报告中的训练指纹保留。公开脚本仅作通用路径调整，新复现使用新指纹，不伪造原检查点身份。',
        'source_transformations':changes,'omitted':['models','checkpoints','logs','third_party','private environment manifests','maintenance receipts','inference caches'],
        'methods':{'v1':'4000G，执行器验证，原开发成绩','v2':'3000G+1000V，独立朴素V','improvement_verify_v1':'基于v2复用G，仅替换V；主线T5-base，四模型补充'}}
    (destination/'manifests/public_release.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    audit=audit_release(destination)
    records=[]
    for p in sorted(destination.rglob('*')):
        if p.is_file():records.append({'path':str(p.relative_to(destination)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())})
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|') as archive:
            for item in records:archive.add(destination/item['path'],arcname=item['path'],recursive=False)
    report={'file_count':len(records),'total_bytes':sum(x['bytes'] for x in records),'gzip_estimate_bytes':len(buffer.getvalue()),'audit':audit,'files':records}
    destination.with_name(destination.name+'.selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--output',type=Path);g.add_argument('--audit',type=Path)
    a=p.parse_args();report=export(a.output) if a.output else audit_release(a.audit)
    print(json.dumps({k:v for k,v in report.items() if k!='files'},ensure_ascii=False,indent=2))
