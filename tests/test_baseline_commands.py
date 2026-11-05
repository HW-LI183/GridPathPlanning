"""四模型入口对称性、下载网络隔离与缺失文件提示；不联网、不训练。"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from gridpath.training_config import ROOT,model_keys,config_for


def fake_workspace(tmp_path):
    (tmp_path/'scripts').mkdir()
    shutil.copy2(ROOT/'scripts/run_phase4_v1.sh',tmp_path/'scripts/run_phase4_v1.sh')
    conda=tmp_path/'fake_conda'
    conda.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['COMMAND_TRACE'],'a') as f:
 f.write(json.dumps({'args':sys.argv[1:],'proxy':os.environ.get('HTTPS_PROXY')})+'\\n')
''')
    conda.chmod(0o755)
    trace=tmp_path/'trace.jsonl'
    env={**os.environ,'GRIDPATH_CONDA':str(conda),'COMMAND_TRACE':str(trace),'HTTPS_PROXY':'http://127.0.0.1:7899'}
    return trace,env


@pytest.mark.parametrize('model',model_keys())
def test_each_model_has_same_train_interface(tmp_path,model):
    trace,env=fake_workspace(tmp_path)
    subprocess.run(['bash','scripts/run_phase4_v1.sh','train',model],cwd=tmp_path,env=env,check=True,capture_output=True)
    calls=[r['args'] for r in map(json.loads,trace.read_text().splitlines())]
    assert len(calls)==3
    assert calls[0][-3:]==['preflight','--models',model]
    assert calls[1][-3:]==['pilot','--model',model]
    assert calls[2][-3:]==['train','--model',model]


@pytest.mark.parametrize('model',model_keys())
def test_each_model_prepare_only_downloads_with_proxy_removed(tmp_path,model):
    trace,env=fake_workspace(tmp_path)
    subprocess.run(['bash','scripts/run_phase4_v1.sh','prepare',model],cwd=tmp_path,env=env,check=True,capture_output=True)
    calls=[json.loads(r) for r in trace.read_text().splitlines()]
    assert len(calls)==1
    assert calls[0]['args'][-3:]==['scripts/fetch_pinned_models.py','--model',model]
    assert calls[0]['proxy'] is None


@pytest.mark.parametrize('model',model_keys())
def test_missing_weights_points_to_matching_prepare_command(tmp_path,monkeypatch,model):
    from gridpath import training_config as tc
    cfg=config_for(model)
    name='flan_models.json' if model.startswith('flan_') else 'models.json'
    (tmp_path/'manifests').mkdir()
    (tmp_path/'manifests'/name).write_text(json.dumps({'models':[{
        'repo_id':cfg['model_id'],'revision':cfg['revision'],'local_directory':cfg['model_directory'],
        'files':[{'name':'model.safetensors','sha256':'not-downloaded'}]}]}))
    monkeypatch.setattr(tc,'ROOT',tmp_path)
    monkeypatch.setattr(tc,'config_for',lambda _:cfg)
    monkeypatch.setattr(tc,'frozen_data',lambda:([],[]))
    with pytest.raises(FileNotFoundError,match='prepare '+model):tc.binding_for(model)


@pytest.mark.parametrize('model',model_keys())
def test_downloader_selects_only_requested_model_and_preserves_manifest(tmp_path,monkeypatch,model):
    import importlib.util
    import sys
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('fetch_models_test',ROOT/'scripts/fetch_pinned_models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for filename,keys in [('models.json',model_keys()[:2]),('flan_models.json',model_keys()[2:])]:
        (tmp_path/filename).write_text(json.dumps({'endpoint':'https://hf-mirror.com','models':[
            {'local_directory':config_for(k)['model_directory'],'repo_id':config_for(k)['model_id'],
             'files':[{'name':'model.safetensors'}]} for k in keys]}))
    original={p.name:p.read_bytes() for p in tmp_path.glob('*.json')}
    monkeypatch.setattr(module,'MANIFEST',tmp_path/'models.json')
    calls=[]
    monkeypatch.setattr(module,'download',lambda endpoint,entry,artifact:calls.append(entry['repo_id']))
    monkeypatch.setattr(sys,'argv',['fetch_pinned_models.py','--model',model])
    module.main()
    assert calls==[config_for(model)['model_id']]
    assert {p.name:p.read_bytes() for p in tmp_path.glob('*.json')}==original
