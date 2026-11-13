"""独立样本≤200；派生验证和共享方法输入只计调用，不重复占样本配额。"""
from dataclasses import asdict
from ..training_config import read_json,write_json,digest
from ..inference import Prediction


class RecordLedger:
    def __init__(self,slots,directory,binding,samples=None):
        if len({r['record_id'] for r in slots})!=len(slots):raise ValueError('调用记录ID重复')
        root=lambda r:r.get('independent_sample_id',r.get('source_sample_id',r['record_id']))
        if samples is None:
            self.samples={root(r):{'sample_id':root(r),'task':r.get('task')} for r in slots}
        else:
            self.samples={r['sample_id']:r for r in samples}
            if len(self.samples)!=len(samples):raise ValueError('独立样本ID重复')
        if len(self.samples)>200:raise ValueError('独立样本超过200')
        if any(root(r) not in self.samples for r in slots):raise ValueError('调用没有已注册的独立父样本')
        self.slots={r['record_id']:r for r in slots};self.directory=directory;self.binding=binding

    def path(self,key):
        if key not in self.slots:raise ValueError('未预留的任务输入，禁止调用模型')
        return self.directory/(digest(key)+'.json')

    def get(self,key):
        p=self.path(key)
        if not p.exists():return None
        r=read_json(p)
        if r['slot']!=self.slots[key] or r['fingerprint']!=self.binding:raise ValueError('任务记录身份变化')
        return r

    def put(self,key,value):
        result={'slot':self.slots[key],'fingerprint':self.binding,**value};old=self.get(key)
        if old is not None and old!=result:raise ValueError('拒绝覆盖不同模型输入—输出记录')
        write_json(self.path(key),result);return result

    def generation(self,key,prompt,target,predict):
        self.path(key)
        if self.slots[key]['task'] not in ('GENERATE','REPAIR'):raise ValueError('验证任务ID不得用于生成')
        old=self.get(key)
        if old:
            expected=target() if callable(target) else target
            if old['status']!='materialized' or old['input_text']!=prompt or old['target_text']!=expected:raise ValueError('生成/修复输入改变')
            return Prediction(**old['prediction'])
        output=predict(prompt)
        target=target() if callable(target) else target
        self.put(key,{'status':'materialized','input_text':prompt,'target_text':target,'prediction':asdict(output)})
        return output

    def verification(self,key,prompt,target,predict):
        self.path(key)
        if self.slots[key]['task']!='VERIFY':raise ValueError('不是验证任务ID')
        old=self.get(key)
        if old:
            expected=target() if callable(target) else target
            if old['status']!='materialized' or old['input_text']!=prompt or old['target_text']!=expected:raise ValueError('验证输入改变')
            return Prediction(**old['prediction']),old['detail']
        output,detail=predict()
        target=target() if callable(target) else target
        # target仅供离线评分，不传入predict闭包或验证器。
        self.put(key,{'status':'materialized','input_text':prompt,'target_text':target,'prediction':asdict(output),'detail':detail})
        return output,detail

    def skip(self,key,status):
        self.path(key)
        if self.slots[key]['task']!='REPAIR':raise ValueError('只允许跳过条件修复任务')
        self.put(key,{'status':'skipped','skip_reason':status,'input_text':None,'target_text':None})

    def audit(self,complete=True):
        from collections import Counter
        files=list(self.directory.glob('*.json')) if self.directory.exists() else []
        for p in files:
            key=read_json(p)['slot']['record_id']
            if key not in self.slots or p!=self.path(key):raise ValueError('发现未注册的额外任务记录')
        records=[self.get(k) for k in self.slots];present=[r for r in records if r]
        if complete and len(present)!=len(self.slots):raise ValueError('有任务尚未运行或登记跳过')
        materialized=[r for r in present if r['status']=='materialized']
        if any(not isinstance(r['input_text'],str) or not isinstance(r['target_text'],str) for r in materialized):raise ValueError('真实任务缺少输入或目标')
        return {'status':'passed','independent_samples':len(self.samples),'independent_sample_limit':200,
                'independent_tasks':dict(Counter(x['task'] for x in self.samples.values())),
                'planned_records':len(self.slots),'materialized_records':len(materialized),
                'skipped_records':len(present)-len(materialized),'model_input_limit':len(self.slots),
                'counting_policy':'独立G/V/R样本配额与派生/共享方法模型输入分开统计',
                'planned_tasks':dict(Counter(x['task'] for x in self.slots.values())),
                'materialized_tasks':dict(Counter(x['slot']['task'] for x in materialized))}
