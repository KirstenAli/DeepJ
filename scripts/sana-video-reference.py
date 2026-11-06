"""Independent numerical fixtures for the Java SANA-Video port.

Requires torch 2.8.0, diffusers 0.42.1, and transformers 5.22.1. Python is used only to regenerate tests.
"""
from pathlib import Path
import json
import torch
from safetensors.torch import save_file
from diffusers import SanaVideoTransformer3DModel, AutoencoderKLWan, DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parents[1] / 'src/test/resources/sana-video'
ROOT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(4917)
torch.set_num_threads(1)
fixtures = {}


def store(name, model):
    path = ROOT / name
    path.mkdir(exist_ok=True)
    model.save_config(path)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, path / 'model.safetensors')


with torch.no_grad():
    model = SanaVideoTransformer3DModel(in_channels=4, out_channels=4,
        num_attention_heads=2, attention_head_dim=12, num_layers=2,
        num_cross_attention_heads=2, cross_attention_head_dim=12, cross_attention_dim=24,
        caption_channels=16, mlp_ratio=2, patch_size=(1,2,2), sample_size=2,
        guidance_embeds=False, qk_norm='rms_norm_across_heads', rope_max_seq_len=64).eval()
    latent, text = torch.randn(1,4,3,4,6), torch.randn(1,7,16)
    mask = torch.tensor([[1,1,1,1,1,0,0]])
    output = model(latent, encoder_hidden_states=text, encoder_attention_mask=mask,
                   timestep=torch.tensor([733])).sample
    store('transformer', model)
    fixtures.update({'transformer.input':latent, 'transformer.text':text, 'transformer.mask':mask.float(),
                     'transformer.output':output})
    transformer = model
    # Wan 2.1 geometry, reduced channel count, all temporal cache paths retained.
    vae = AutoencoderKLWan(base_dim=4, z_dim=4, dim_mult=[1,2,4,4], num_res_blocks=1,
        attn_scales=[], temperal_downsample=[False,True,True], scale_factor_temporal=4,
        scale_factor_spatial=8, latents_mean=[0]*4, latents_std=[1]*4).eval()
    store('vae',vae)
    for time in (1,2,3,4):
        latent = torch.randn(1,4,time,2,3)
        fixtures[f'vae{time}.input'] = latent
        fixtures[f'vae{time}.output'] = vae.decode(latent).sample.contiguous()

with torch.enable_grad():
    latent=fixtures['transformer.input'].clone().requires_grad_()
    text=fixtures['transformer.text'].clone().requires_grad_()
    result=transformer(latent,encoder_hidden_states=text,encoder_attention_mask=mask,
        timestep=torch.tensor([733])).sample
    result.square().mean().backward()
    fixtures['transformer.dx']=latent.grad
    fixtures['transformer.dt']=text.grad
    for name in ('patch_embedding.weight','transformer_blocks.0.ff.conv_temp.weight',
                 'transformer_blocks.0.attn1.to_q.weight'):
        fixtures['transformer.grad.'+name]=transformer.get_parameter(name).grad
    latent=fixtures['vae2.input'].clone().requires_grad_()
    vae.decode(latent).sample.square().mean().backward()
    fixtures['vae.dx']=latent.grad
    fixtures['vae.dw']=vae.decoder.conv_in.weight.grad

for prefix, causal, kernel in [('conv',True,(3,3,3)),('point',True,(1,1,1)),('patch',False,(1,2,2))]:
    x = torch.randn(1,2,3,4,6,requires_grad=True)
    w = torch.randn(3,2,*kernel,requires_grad=True)
    b = torch.randn(3,requires_grad=True)
    inp = torch.nn.functional.pad(x,(1,1,1,1,2,0)) if kernel==(3,3,3) else x
    y = torch.nn.functional.conv3d(inp,w,b,stride=(1,2,2) if prefix=='patch' else 1)
    y.square().sum().backward()
    fixtures.update({prefix+'.'+k:v.detach().contiguous() for k,v in
        {'input':x,'weight':w,'bias':b,'output':y,'dx':x.grad,'dw':w.grad,'db':b.grad}.items()})

for count in (1,2,5,15,20,50):
    scheduler = DPMSolverMultistepScheduler(algorithm_type='dpmsolver++',solver_order=2,
        solver_type='midpoint',prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=8)
    scheduler.set_timesteps(count)
    sample=torch.randn(1,4,2,2,3)
    fixtures[f'solver{count}.times']=scheduler.timesteps.float()
    fixtures[f'solver{count}.sigmas']=scheduler.sigmas
    fixtures[f'solver{count}.input']=sample
    for i,time in enumerate(scheduler.timesteps):
        flow=torch.randn_like(sample)
        fixtures[f'solver{count}.{i}.flow']=flow
        sample=scheduler.step(flow,time,sample).prev_sample
        fixtures[f'solver{count}.{i}.output']=sample
    scheduler.save_config(ROOT/'scheduler')
save_file({k:v.detach().clone().contiguous() for k,v in fixtures.items()}, ROOT/'model.safetensors')
(ROOT/'reference.json').write_text(json.dumps({'torch':torch.__version__,'diffusers':'0.42.1','seed':4917},indent=2))
print('Fixtures saved:',ROOT)

# Complete, deliberately tiny pipeline for download-free end-to-end CLI tests.
import shutil
from transformers import Gemma2Config, Gemma2Model
full = ROOT / 'pipeline'
full.mkdir(exist_ok=True)
(full/'model_index.json').write_text(json.dumps({'_class_name':'SanaVideoPipeline'}))
shutil.copytree(ROOT.parent/'sana/pipeline/tokenizer',full/'tokenizer',dirs_exist_ok=True)
shutil.copytree(ROOT/'scheduler',full/'scheduler',dirs_exist_ok=True)
text = Gemma2Model(Gemma2Config(vocab_size=32,hidden_size=16,intermediate_size=24,
    num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
    max_position_embeddings=4096,sliding_window=4096,query_pre_attn_scalar=8,
    hidden_activation='gelu_pytorch_tanh')).eval()
video = SanaVideoTransformer3DModel(in_channels=16,out_channels=16,num_attention_heads=2,
    attention_head_dim=12,num_layers=1,num_cross_attention_heads=2,cross_attention_head_dim=12,
    cross_attention_dim=24,caption_channels=16,mlp_ratio=2,patch_size=(1,2,2),sample_size=2,
    guidance_embeds=False,qk_norm='rms_norm_across_heads').eval()
vae = AutoencoderKLWan(base_dim=2,z_dim=16,dim_mult=[1,2,2,2],num_res_blocks=0,
    attn_scales=[],temperal_downsample=[False,True,True],scale_factor_temporal=4,scale_factor_spatial=8,
    latents_mean=[0]*16,latents_std=[1]*16).eval()
for name,model in (('text_encoder',text),('transformer',video),('vae',vae)):
    path=full/name;path.mkdir(exist_ok=True)
    if name=='text_encoder':
        config=model.config.to_dict()
        config['rope_theta']=10000.0
        (path/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    else:model.save_config(path)
    save_file({k:v.contiguous() for k,v in model.state_dict().items()},path/'model.safetensors')

# Optional tokenizer oracle from the official local checkpoint (no runtime downloads).
import argparse, inspect
parser=argparse.ArgumentParser()
parser.add_argument('--tokenizer',type=Path)
args=parser.parse_args()
if args.tokenizer:
    from transformers import AutoTokenizer
    from diffusers.pipelines.sana_video.pipeline_sana_video import SanaVideoPipeline
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    tokenizer.padding_side='right'
    prefix='\n'.join(inspect.signature(SanaVideoPipeline.__call__).parameters['complex_human_instruction'].default)
    cases=[]
    for positive,text in ((True,'A red fox walking through a snowy forest.'),(False,'')):
        length=len(tokenizer.encode(prefix))+298 if positive else 300
        encoded=tokenizer((prefix if positive else '')+text.lower().strip(),padding='max_length',
            max_length=length,truncation=True)
        selected=[0]+list(range(length-299,length))
        cases.append({'positive':positive,'text':text,'ids':encoded.input_ids,'valid':encoded.attention_mask,
            'selected':selected,'mask':[encoded.attention_mask[i] for i in selected]})
    (ROOT/'tokenizer-reference.json').write_text(json.dumps({'cases':cases},indent=2))
