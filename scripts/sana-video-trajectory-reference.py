"""Coupled denoising and decoder reference using the small SANA-Video fixtures.

Run sana-video-reference.py first. All computations run on CPU; no pretrained
weights are downloaded. Java reads these exact inputs, weights, and outputs.
"""
from pathlib import Path
import json
import torch
import diffusers
from safetensors.torch import load_file, save_file
from diffusers import SanaVideoTransformer3DModel, AutoencoderKLWan, DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parents[1] / 'src/test/resources/sana-video'
torch.set_num_threads(1)
source = load_file(ROOT / 'model.safetensors')
fixtures = {}
(ROOT / 'trajectory').mkdir(exist_ok=True)
positive = source['transformer.text']
negative = positive.flip(-1) * 0.5
positive_mask = source['transformer.mask'].long()
negative_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0]])
fixtures.update({'input': source['transformer.input'], 'positive': positive, 'negative': negative,
                 'positive_mask': positive_mask.float(), 'negative_mask': negative_mask.float()})
vae = AutoencoderKLWan.from_config(json.loads((ROOT / 'vae/config.json').read_text())).eval()
vae.load_state_dict(load_file(ROOT / 'vae/model.safetensors'))
with torch.no_grad():
    model = SanaVideoTransformer3DModel.from_config(json.loads((ROOT / 'transformer/config.json').read_text())).eval()
    model.load_state_dict(load_file(ROOT / 'transformer/model.safetensors'))
    # Preserve the rotary buffers at their constructor precision, as checkpoint loading does.
    for count in (20, 50):
        name = f'float32.{count}'
        sample = fixtures['input'].clone()
        scheduler = DPMSolverMultistepScheduler.from_pretrained(ROOT / 'scheduler', local_files_only=True)
        scheduler.set_timesteps(count)
        fixtures[name + '.times'] = scheduler.timesteps.float()
        for index, time in enumerate(scheduler.timesteps):
            inputs = dict(hidden_states=sample, timestep=time.reshape(1))
            conditioned = model(**inputs, encoder_hidden_states=positive,
                                encoder_attention_mask=positive_mask).sample.float()
            unconditioned = model(**inputs, encoder_hidden_states=negative,
                                  encoder_attention_mask=negative_mask).sample.float()
            flow = unconditioned + 6 * (conditioned - unconditioned)
            sample = scheduler.step(flow, time, sample).prev_sample
            fixtures[f'{name}.{index}.sample'] = sample.clone()
        fixtures[name + '.pixels'] = vae.decode(sample).sample.contiguous()
        print(name, 'complete', flush=True)
    save_file({name: tensor.detach().clone().contiguous() for name, tensor in fixtures.items()},
              ROOT / 'trajectory/model.safetensors')
(ROOT / 'trajectory-reference.json').write_text(json.dumps({
    'torch': torch.__version__, 'diffusers': diffusers.__version__, 'device': 'cpu',
    'guidance': 6, 'steps': [20, 50], 'latent_shape': list(fixtures['input'].shape),
    'purpose': 'Coupled feedback trajectory and causal video decoder, identical saved inputs and weights'
}, indent=2) + '\n')
