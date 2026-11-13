"""方法适配：基线分支不导入结构化模型，不读取另一方法的产物。"""
from ..evaluation import Predictor
from ..v2.prompts import render_verification


def prompt(method,grid,candidate):
    if method=='v2':return render_verification(grid,candidate)
    from ..planning.protocol import render
    return render(grid,candidate)


class PlainVerifier:
    def __init__(self,model,tokenizer,cfg,budget=None,cache=None):
        self.predictor=Predictor(model,tokenizer,cfg,budget,cache)
    @property
    def calls(self):return self.predictor.calls
    @property
    def hits(self):return self.predictor.hits
    def __call__(self,grid,candidate):
        p=self.predictor(render_verification(grid,candidate))
        return p,{'verifier_kind':'plain_seq2seq','executor_used':False,'raw_output':p.text}


def validate(method,candidate,prediction,detail):
    if prediction.failure is not None:return
    if detail.get('executor_used') is not False:raise ValueError('验证必须来自模型预测')
    if method=='v2':
        if detail.get('verifier_kind')!='plain_seq2seq' or detail.get('raw_output')!=prediction.text:raise ValueError('朴素V文本被替换')
    else:
        from ..planning.protocol import aggregate,actions_of
        if prediction.text!=aggregate(detail['tags'],len(actions_of(candidate))):raise ValueError('诊断不是模型逐步预测的聚合')
