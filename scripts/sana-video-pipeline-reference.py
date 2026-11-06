"""Full tiny pipeline oracle from Java-exported token IDs and noise, using CPU only."""
from pathlib import Path
import json
import torch
from safetensors.torch import load_file, save_file
from transformers import Gemma2Config, Gemma2Model
from diffusers import SanaVideoTransformer3DModel, AutoencoderKLWan, DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parents[1] / 'src/test/resources/sana-video'
MODEL = ROOT / 'pipeline'
torch.set_num_threads(1)
inputs = json.loads((ROOT / 'pipeline-inputs.json').read_text())
vae = AutoencoderKLWan.from_config(json.loads((MODEL / 'vae/config.json').read_text())).eval()
vae.load_state_dict(load_file(MODEL / 'vae/model.safetensors'))
fixtures = {}
with torch.no_grad():
    encoder = Gemma2Model(Gemma2Config.from_pretrained(MODEL / 'text_encoder')).eval()
    encoder.load_state_dict(load_file(MODEL / 'text_encoder/model.safetensors'))
    encoder.set_attn_implementation('eager')
    for name in ('positive', 'negative'):
        caption = inputs[name]
        states = encoder(torch.tensor([caption['tokens']]), attention_mask=torch.tensor([caption['valid']]))
        fixtures['float32.' + name] = states.last_hidden_state[:, caption['selected']].float()
    model = SanaVideoTransformer3DModel.from_config(json.loads((MODEL / 'transformer/config.json').read_text()))
    model.load_state_dict(load_file(MODEL / 'transformer/model.safetensors'))
    # The checkpoint dtype applies to weights, not the generated rotary buffers.
    model.eval()
    sample = torch.tensor(inputs['noise']).reshape(1, 16, 3, 2, 2)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL / 'scheduler', local_files_only=True)
    scheduler.set_timesteps(inputs['steps'])
    for index, time in enumerate(scheduler.timesteps):
        flows = {}
        for name in ('positive', 'negative'):
            flows[name] = model(sample, timestep=time.reshape(1),
                encoder_hidden_states=fixtures['float32.' + name],
                encoder_attention_mask=torch.tensor([inputs[name]['mask']])).sample.float()
        flow = flows['negative'] + inputs['guidance'] * (flows['positive'] - flows['negative'])
        fixtures[f'float32.flow.{index}'] = flow.clone()
        sample = scheduler.step(flow, time, sample).prev_sample
        fixtures[f'float32.sample.{index}'] = sample.clone()
    mean = torch.tensor(vae.config.latents_mean).reshape(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).reshape(1, 16, 1, 1, 1)
    fixtures['float32.pixels'] = vae.decode(sample * std + mean).sample
    print('float32', 'full pipeline complete', flush=True)
(ROOT / 'pipeline-reference').mkdir(exist_ok=True)
save_file({name: value.clone().contiguous() for name, value in fixtures.items()},
          ROOT / 'pipeline-reference/model.safetensors')
