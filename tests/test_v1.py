"""v1数据配额、修复边界、分母及训练暴露/续跑验证；不使用正式模型。"""
import json
from dataclasses import asdict

import pytest

from gridpath.data_schema import grid_to_dict
from gridpath.evaluation import evaluate_case,repair_once,summarize_panel
from gridpath.inference import Prediction
from gridpath.training_config import ROOT,read_rows,read_json,config_for
from gridpath.types import Grid


@pytest.fixture
def grid():return Grid(3,3,frozenset({(0,1)}),(1,0),(1,2))


def row(grid,candidate=None):
    return {**grid_to_dict(grid),"task":"REPAIR" if candidate is not None else "GENERATE",
        "sample_id":"sample","candidate_plan":candidate,"shortest_distance":2,
        "reference_plan":["right","right"],"input_text":"unused","target_text":"ORACLE"}


def test_frozen_data_is_g4000_dev_g100_r100_and_unique_environments():
    from collections import Counter
    train=read_rows(ROOT/"data/v1/train.jsonl");dev=read_rows(ROOT/"data/v1/dev.jsonl")
    assert Counter(r["task"] for r in train)=={"GENERATE":4000}
    assert Counter(r["task"] for r in dev)=={"GENERATE":100,"REPAIR":100}
    assert len({r["environment_id"] for r in train})==4000
    assert len({r["environment_id"] for r in dev})==200
    repair=[r for r in dev if r["task"]=="REPAIR"]
    assert Counter(r["error_reason"] or "valid" for r in repair)=={
        "obstacle_collision":25,"out_of_bounds":25,"invalid_action_token":25,"goal_not_reached":25}
    assert all(r["is_valid"] is False and len(r["edit_operations"])==1 for r in repair)
    assert not {r["layout_id"] for r in train}&{r["layout_id"] for r in dev}


@pytest.mark.parametrize("change",["config","sources","train","test","generate"])
def test_repair_only_lineage_rejects_changes_to_training_or_selection(change):
    from copy import deepcopy
    from gridpath.training_config import validate_training_lineage,digest
    generated=[{"task":"GENERATE","sample_id":"fixed"}]
    original={"config":{"lr":1e-4},"model":{"revision":"fixed"},"packages":{"torch":"fixed"}}
    payload={**deepcopy(original),"sources":{"runtime":"approved"}}
    lineage={"repair_dev_revision":"invalid100","models":{"model":{"training_binding":original}},
        "approved_runtime_sources":{"runtime":"approved"},"unchanged_files":{"train":"trainhash","test_id":"testhash"},
        "dev_generate_sha256":digest(generated)}
    manifest={"files":{"train":{"sha256":"trainhash"},"test_id":{"sha256":"testhash"}}}
    validate_training_lineage(lineage,"model",payload,manifest,generated)
    if change=="config":payload["config"]["lr"]=1e-3
    elif change=="sources":payload["sources"]["runtime"]="changed"
    elif change=="train":manifest["files"]["train"]["sha256"]="changed"
    elif change=="test":manifest["files"]["test_id"]["sha256"]="changed"
    else:generated[0]["sample_id"]="changed"
    with pytest.raises(ValueError):validate_training_lineage(lineage,"model",payload,manifest,generated)


@pytest.mark.parametrize("model",["t5_base","t5_small"])
def test_same_training_recipe_and_exact_budget(model):
    cfg=config_for(model)
    assert cfg["max_updates"]*cfg["effective_batch_size"]==4000*3
    assert cfg["learning_rate"]==1e-4 and cfg["warmup_updates"]==12


def test_prefix_uses_last_executed_state_and_exact_generation_template(grid):
    from gridpath.prompts import render_generation
    seen=[]
    def predict(text):seen.append(text);return Prediction("right",2)
    final,details=repair_once(grid,Prediction("right up right",4),"prefix",predict)
    assert seen==[render_generation(Grid(3,3,grid.obstacles,(1,1),grid.goal))]
    assert final.text=="right right" and details["prefix"]==["right"]


def test_early_stop_keeps_entire_prefix_and_suffix_failure_is_not_cleaned(grid):
    output,details=repair_once(grid,Prediction("right",2),"prefix",lambda _:Prediction("right",1,"token_limit"))
    assert output.text=="right right" and output.failure=="token_limit"
    assert details["prefix"]==["right"]


@pytest.mark.parametrize("method",["prompt","prefix"])
def test_valid_and_valid_text_interface_failure_preserved_without_hidden_retry(grid,method):
    original=Prediction("right right",2,"token_limit")
    def never(_):pytest.fail("有效文本不能悄悄重生成为接口成功")
    result,detail=repair_once(grid,original,method,never)
    assert result==original and detail["model_calls"]==0


def test_prefix_at_goal_is_separated_from_neural_success(grid):
    r=row(grid,["right","right","right"])
    def predict(_):return Prediction("up",2)
    case=evaluate_case(r,predict,config_for("t5_base"))
    report=summarize_panel([r],[case],config_for("t5_base"))
    prefix=report["repairs"]["prefix"]
    assert prefix["repair_success_rate"]==1
    assert prefix["empty_suffix_success_count"]==1 and prefix["neural_suffix_success_count"]==0
    assert prefix["logical_repair_model_calls"]==0


def test_gained_denominator_and_interface_failure_in_full_panel(grid):
    r=row(grid,["right","up"])
    calls=[]
    def predict(text):
        calls.append(text)
        return Prediction("right right" if text.startswith("REPAIR") else "right",2)
    case=evaluate_case(r,predict,config_for("t5_base"))
    s=summarize_panel([r],[case],config_for("t5_base"))
    assert s["direct"]["success_rate"]==0
    assert s["repairs"]["prompt"]["repair_success_rate"]==1
    assert s["repairs"]["prefix"]["repair_success_rate"]==1
    assert all("ORACLE" not in text for text in calls)
    assert s["by_group"]["original_error"]["obstacle_collision"]["count"]==1


def test_resume_refuses_uncommitted_exposure_and_fourth_epoch():
    from gridpath.training import validate_state
    cfg=config_for("t5_base")
    state={"global_step":10,"consumed_examples":40,"evaluated_steps":[]}
    validate_state(state,{"reserved_examples":40,"completed_updates":10},cfg)
    with pytest.raises(RuntimeError):validate_state(state,{"reserved_examples":44,"completed_updates":10},cfg)
    with pytest.raises(ValueError):validate_state({"global_step":3001,"consumed_examples":12004,"evaluated_steps":[1000,2000,3000]},
        {"reserved_examples":12004,"completed_updates":3001},cfg)


def test_checkpoint_selection_ignores_train_and_repair_results():
    from gridpath.training import selection_key
    report={"metrics":{"success_rate":.6,"optimal_success_rate":.5},"nll":1.}
    assert selection_key(report)==[.6,.5,-1.]


def test_entrypoint_sets_cublas_before_torch_import():
    import os,subprocess,sys
    env=dict(os.environ);env.pop("CUBLAS_WORKSPACE_CONFIG",None)
    code='''
import builtins,runpy,os
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name=="torch" or name.startswith("torch."):
  assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8"
 return original(name,*args,**kwargs)
builtins.__import__=guarded
runpy.run_path("scripts/phase4.py",run_name="import_only")
assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"
'''
    subprocess.run([sys.executable,"-c",code],env=env,cwd=ROOT,check=True,capture_output=True,timeout=30)


@pytest.mark.parametrize("projection",["relu","gated-gelu"])
def test_real_cpu_training_pilot_resume_matches_uninterrupted_three_passes(tmp_path,monkeypatch,projection):
    import torch
    import safetensors.torch
    from transformers import T5Config,T5ForConditionalGeneration,AutoTokenizer,AutoModelForSeq2SeqLM
    from gridpath import training,evaluation
    from gridpath.training_helpers import validate_checkpoint
    cfg={**config_for("t5_small"),"max_updates":6,"evaluate_steps":[2,4,6],
         "pilot_updates":1,"checkpoint_every_updates":2,"run_directory":"checkpoints/v1/tiny"}
    features=[{"sample_id":f"s{i}","input_text":"grid","target_text":"right","task":"GENERATE","obstacles":[[0,0]]} for i in range(8)]
    dev=[{**features[0],"sample_id":"d0"},{**features[1],"sample_id":"d1"}]
    binding={"fingerprint":"tiny_cpu_test_only","config":cfg}
    class TinyTokenizer:
        pad_token_id=0
        def __call__(self,text):return {"input_ids":[5,1]}
        def save_pretrained(self,path):(path/"tokenizer_config.json").write_text('{}')
    def factory(*args,**kwargs):
        return T5ForConditionalGeneration(T5Config(vocab_size=16,d_model=8,d_ff=16,d_kv=4,
            num_layers=1,num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1,feed_forward_proj=projection))
    module_to=torch.nn.Module.to;tensor_to=torch.Tensor.to
    def to_cpu(original):
        def adapted(self,*args,**kwargs):
            if args and args[0]=="cuda":args=("cpu",*args[1:])
            return original(self,*args,**kwargs)
        return adapted
    actual_load=safetensors.torch.load_model
    def load_cpu(model,path,**kwargs):return actual_load(model,path,strict=kwargs.get("strict",True),device="cpu")
    monkeypatch.setattr(torch.nn.Module,"to",to_cpu(module_to))
    monkeypatch.setattr(torch.Tensor,"to",to_cpu(tensor_to))
    monkeypatch.setattr(safetensors.torch,"load_model",load_cpu)
    monkeypatch.setattr(AutoModelForSeq2SeqLM,"from_pretrained",factory)
    monkeypatch.setattr(AutoTokenizer,"from_pretrained",lambda *a,**k:TinyTokenizer())
    monkeypatch.setattr(training,"binding_for",lambda *a,**k:(cfg,features,dev,binding))
    monkeypatch.setattr(training,"require_gpu",lambda _:None)
    monkeypatch.setattr(evaluation,"evaluate_generation",lambda model,tok,rows,*args:{
        "count":len(rows),"metrics":{"success_rate":0.,"optimal_success_rate":0.},"nll":1.})
    before=torch.get_num_threads();torch.set_num_threads(1)
    try:
        weights=[]
        for name,split in (("full",False),("resumed",True)):
            root=tmp_path/name;root.mkdir();monkeypatch.setattr(training,"ROOT",root)
            if split:
                pilot=training.run_training("t5_small",pilot=True)
                assert pilot["state"]["global_step"]==1 and pilot["state"]["consumed_examples"]==4
            result=training.run_training("t5_small")
            assert result["state"]["global_step"]==6 and result["state"]["consumed_examples"]==24
            assert result["state"]["evaluated_steps"]==[2,4,6]
            run=root/cfg["run_directory"]
            validate_checkpoint(run/"latest",binding["fingerprint"])
            weights.append(safetensors.torch.load_file(run/"latest/model.safetensors"))
            entries=len(read_json(root/"manifests/gpu_budget.json")["entries"])
            assert training.run_training("t5_small")["status"]=="complete"
            assert len(read_json(root/"manifests/gpu_budget.json")["entries"])==entries
        assert weights[0].keys()==weights[1].keys()
        assert all(torch.equal(weights[0][k],weights[1][k]) for k in weights[0])
    finally:torch.set_num_threads(before)


def test_tiny_random_cpu_inference_cache_and_source_guard(tmp_path):
    import torch
    from transformers import AutoTokenizer,T5Config,T5ForConditionalGeneration
    from gridpath.evaluation import Predictor
    from gridpath.prompts import render_generation
    before=torch.get_num_threads();torch.set_num_threads(1)
    try:
        tok=AutoTokenizer.from_pretrained(ROOT/"models/t5-small",local_files_only=True)
        model=T5ForConditionalGeneration(T5Config(vocab_size=len(tok),d_model=8,d_ff=16,d_kv=4,
            num_layers=1,num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1)).eval()
        cfg={**config_for("t5_small"),"generation_tokens":8}
        predictor=Predictor(model,tok,cfg,cache=tmp_path/"cache")
        text=render_generation(Grid(3,3,frozenset(),(0,0),(2,2)))
        output=predictor(text)
        assert output==predictor(text) and predictor.calls==1 and predictor.hits==1
        guarded=Predictor(model,tok,{**cfg,"max_source_tokens":1})
        assert guarded(text).failure=="source_limit" and guarded.calls==0
    finally:torch.set_num_threads(before)


@pytest.mark.parametrize('key',['t5_base','t5_small','flan_t5_base','flan_t5_small'])
def test_four_baselines_share_recipe_but_isolate_checkpoints(key):
    cfg=config_for(key)
    assert cfg['run_directory']==f'checkpoints/v1/{key}'
    reference=config_for('t5_base')
    assert {k:v for k,v in cfg.items() if k not in ('model_key','model_id','revision','model_directory','run_directory')}=={
        k:v for k,v in reference.items() if k not in ('model_key','model_id','revision','model_directory','run_directory')}


def test_four_model_selection_preserves_historical_lock():
    from gridpath.training_config import primary_lock_path,model_keys
    assert len(model_keys())==4
    assert primary_lock_path().name=='primary_model_v1_four.json'
    assert primary_lock_path(('t5_base','t5_small')).name=='primary_model_v1.json'
    with pytest.raises(ValueError):primary_lock_path(('flan_t5_base','flan_t5_small'))


def test_flan_model_revisions_match_separate_manifest():
    manifest=read_json(ROOT/'manifests/flan_models.json')
    for key in ('flan_t5_base','flan_t5_small'):
        cfg=config_for(key)
        card=next(c for c in manifest['models'] if c['repo_id']==cfg['model_id'])
        assert card['revision']==cfg['revision']
        assert card['local_directory']==cfg['model_directory']
        assert next(f for f in card['files'] if f['name']=='model.safetensors')['sha256']


@pytest.mark.parametrize('flan_score,expected',[(.7,'flan_t5_base'),(.6,'t5_base')])
def test_four_model_primary_selection_keeps_old_lock(tmp_path,monkeypatch,flan_score,expected):
    from gridpath import training,training_config
    keys=training_config.model_keys()
    configs={key:config_for(key) for key in keys}
    dataset=tmp_path/'manifests/dataset_manifest.json'
    dataset.parent.mkdir();dataset.write_text('{}')
    old=tmp_path/'configs/primary_model_v1.json'
    old.parent.mkdir();old.write_text('{"historical":"untouched"}')
    for key,score in zip(keys,(.6,.23,flan_score,.4)):
        run=tmp_path/configs[key]['run_directory']
        (run/'evaluations').mkdir(parents=True)
        training_config.write_json(run/'completed.json',{'fingerprint':key,'state':{'consumed_examples':12000,'best_step':1000}})
        training_config.write_json(run/'evaluations/step_1000.json',{
            'fingerprint':key,'dev_generate':{'count':100,'metrics':{'success_rate':score,'optimal_success_rate':score}}})
    monkeypatch.setattr(training,'ROOT',tmp_path)
    monkeypatch.setattr(training_config,'ROOT',tmp_path)
    monkeypatch.setattr(training,'binding_for',lambda key,**kw:(configs[key],[],[],{'fingerprint':key}))
    monkeypatch.setattr(training,'validate_checkpoint',lambda path,fingerprint:{'state':{'best_step':1000},'files':{'model.safetensors':fingerprint}})
    result=training.select_primary()
    assert result['selected']['model']==expected
    assert len(result['candidates'])==4
    assert (tmp_path/'configs/primary_model_v1_four.json').exists()
    assert old.read_text()=='{"historical":"untouched"}'
