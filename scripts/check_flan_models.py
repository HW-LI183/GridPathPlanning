#!/usr/bin/env python3
"""短CPU检查Flan固定文件和各划分token长度；不加载模型权重、不推理。"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gridpath.training_config import ROOT,read_json,read_rows,write_json,sha256
from gridpath.prompts import render_repair
from gridpath.data_schema import grid_from_record
from gridpath.types import ACTION_ORDER


def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-weights',action='store_true')
    args=p.parse_args()
    manifest=read_json(ROOT/'manifests/dataset_manifest.json')
    report={'gpu_calls':0,'models':{}}
    for card in read_json(ROOT/'manifests/flan_models.json')['models']:
        directory=ROOT/card['local_directory']
        weights_present=True
        for file in card['files']:
            path=directory/file['name']
            if not path.exists() and file['name']=='model.safetensors' and not args.require_weights:
                weights_present=False
                continue
            if not file['sha256'] or sha256(path)!=file['sha256']:
                raise ValueError(f'文件不存在或哈希错误:{path}')
        tok=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        stats={}
        for name,item in manifest['files'].items():
            if name=='test_environments':continue
            rows=read_rows(ROOT/item['path'])
            inputs=[r['input_text'] for r in rows]
            inputs += [render_repair(grid_from_record(r),r['candidate_plan'],r['verification_label']) for r in rows if 'candidate_plan' in r]
            source=tok(inputs,truncation=False)['input_ids']
            target=tok([r['target_text'] for r in rows],truncation=False)['input_ids']
            if any(len(x)>512 or tok.unk_token_id in x for x in source) or any(len(x)>256 for x in target):
                raise ValueError(f'长度/未知token检查失败:{card["repo_id"]}/{name}')
            stats[name]={'max_source':max(map(len,source)),'max_target':max(map(len,target))}
        row=max(read_rows(ROOT/manifest['files']['dev']['path']),key=lambda r:len(r['obstacles']))
        worst=max(len(tok(render_repair(grid_from_record(row),[a]*72,'invalid | step=73 | reason=obstacle_collision'))['input_ids']) for a in ACTION_ORDER)
        if worst>512:raise ValueError('72动作修复超长')
        report['models'][card['repo_id']]={'status':'passed','weights_present':weights_present,'token_lengths':stats,'repair_72_actions':worst}
    write_json(ROOT/'results/flan_preparation.json',report)
    print(__import__('json').dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
