"""结构化组件的小型CPU回归；不加载真实权重、不启动正式训练。"""
import torch
from transformers import T5Config,T5ForConditionalGeneration
from gridpath.planning.model import StepVerifier,path_loss
from gridpath.planning.protocol import aggregate,virtual_states


def test_original_seq2seq_encoder_initialization_and_safe_roundtrip(tmp_path):
    cfg=T5Config(vocab_size=40,d_model=16,d_kv=4,d_ff=32,num_layers=1,num_heads=4,dropout_rate=0.)
    original=T5ForConditionalGeneration(cfg);original.save_pretrained(tmp_path/'original',safe_serialization=True)
    verifier=StepVerifier.from_backbone(tmp_path/'original')
    assert torch.equal(original.shared.weight,verifier.encoder.shared.weight)
    inputs={'input_ids':torch.tensor([[2,3,4]]),'attention_mask':torch.ones(1,3,dtype=torch.long),
            'positions':torch.tensor([[0,1,2]]),'kinds':torch.tensor([[0,0,1]])}
    verifier.eval();expected=verifier(**inputs)
    assert expected.shape==(1,3,6)
    assert torch.all(expected[0,:2,4:]<-1000) and torch.all(expected[0,2,:4]<-1000)
    verifier.save_pretrained(tmp_path/'v',safe_serialization=True)
    loaded=StepVerifier.from_pretrained(tmp_path/'v').eval()
    assert torch.equal(loaded(**inputs),expected)


def test_masked_suffix_and_decision_weights():
    logits=torch.zeros((2,4,6),requires_grad=True)
    labels=torch.tensor([[0,1,-100,-100],[0,0,0,4]])
    loss=path_loss(logits,labels);loss.backward()
    assert torch.all(logits.grad[0,2:]==0)
    assert torch.allclose(logits.grad[0,1].abs().sum(),logits.grad[1,3].abs().sum()*4)


def test_unchecked_geometry_and_first_predicted_error_aggregation():
    assert virtual_states((0,0),['left','jump','up'])==[(0,0),(0,-1),(0,-1),(-1,-1)]
    assert aggregate([0,0,1,2,5],4)=='invalid | step=3 | reason=obstacle_collision'
    assert aggregate([0,0,4],2)=='valid'
