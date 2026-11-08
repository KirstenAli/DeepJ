"""Three-step compact video oracle, using the saved tiny checkpoint, token IDs and noise.

Run on CPU with PyTorch 2.8.0, Transformers 5.22.1 and Diffusers 0.42.1.
No pretrained weights are downloaded. Generated rotary buffers, the scheduler
and the Wan decoder stay FP32; checkpoint parameters use the selected dtype.
"""
from pathlib import Path
import json
import sys
import torch
from safetensors.torch import load_file, save_file
from transformers import Gemma2Config, Gemma2Model
from diffusers import SanaVideoTransformer3DModel, AutoencoderKLWan, DPMSolverMultistepScheduler

ROOT = Path(__file__).resolve().parents[1] / 'src/test/resources/sana-video'
MODEL = ROOT / 'pipeline'
torch.set_num_threads(1)
inputs = json.loads((ROOT / 'pipeline-inputs.json').read_text())
fixtures = {}


def split_linear(module, value):
    """Optional rounding control; the normal reference uses unmodified torch.nn.Linear."""
    result = torch.nn.functional.linear(value, module.weight)
    return result if module.bias is None else result + module.bias


if '--split-bias' in sys.argv:
    torch.nn.Linear.forward = split_linear


def weights(model, directory, dtype):
    state = {name: value.to(dtype) for name, value in load_file(directory / 'model.safetensors').items()}
    model.load_state_dict(state, assign=True)
    return model.eval().requires_grad_(False)


def captions(dtype):
    config = Gemma2Config.from_pretrained(MODEL / 'text_encoder')
    config._attn_implementation = 'eager'
    encoder = weights(Gemma2Model(config), MODEL / 'text_encoder', dtype)
    names = ('positive', 'negative')
    length = max(len(inputs[name]['tokens']) for name in names)
    tokens = [inputs[name]['tokens'] + [0] * (length - len(inputs[name]['tokens'])) for name in names]
    valid = [inputs[name]['valid'] + [0] * (length - len(inputs[name]['valid'])) for name in names]
    hidden = encoder(torch.tensor(tokens), attention_mask=torch.tensor(valid)).last_hidden_state
    return {name: hidden[row:row + 1, inputs[name]['selected']] for row, name in enumerate(names)}


def trajectory(name, dtype, vae):
    features = captions(dtype)
    fixtures.update({name + '.' + key: value.float() for key, value in features.items()})
    config = json.loads((MODEL / 'transformer/config.json').read_text())
    model = weights(SanaVideoTransformer3DModel.from_config(config), MODEL / 'transformer', dtype)
    sample = torch.tensor(inputs['noise']).reshape(1, 16, 3, 2, 2)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(MODEL / 'scheduler', local_files_only=True)
    scheduler.set_timesteps(3)
    for index, time in enumerate(scheduler.timesteps):
        flow = prediction(model, sample, features, time, dtype)
        fixtures[f'{name}.flow.{index}'] = flow.clone()
        sample = scheduler.step(flow, time, sample).prev_sample
        fixtures[f'{name}.sample.{index}'] = sample.clone()
    mean = torch.tensor(vae.config.latents_mean).reshape(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).reshape(1, 16, 1, 1, 1)
    fixtures[name + '.pixels'] = vae.decode(sample * std + mean).sample
    print(name, 'three-step pipeline complete', flush=True)


def prediction(model, sample, features, time, dtype):
    names = ('negative', 'positive')
    prediction = model(torch.cat([sample, sample]).to(dtype), timestep=time.expand(2),
        encoder_hidden_states=torch.cat([features[name] for name in names]),
        encoder_attention_mask=torch.tensor([inputs[name]['mask'] for name in names])).sample.float()
    negative, positive = prediction.chunk(2)
    return negative + inputs['guidance'] * (positive - negative)


with torch.inference_mode():
    config = json.loads((MODEL / 'vae/config.json').read_text())
    vae = weights(AutoencoderKLWan.from_config(config), MODEL / 'vae', torch.float32)
    for name, dtype in (('float32', torch.float32), ('float16', torch.float16), ('bfloat16', torch.bfloat16)):
        trajectory(name, dtype, vae)
if not all(torch.isfinite(value).all() for value in fixtures.values()):
    raise RuntimeError('Non-finite compact pipeline reference')
destination = ROOT / 'compact-pipeline-reference'
if '--split-bias' in sys.argv:
    destination = ROOT / 'compact-pipeline-split-bias'
destination.mkdir(exist_ok=True)
save_file({name: value.clone().contiguous() for name, value in fixtures.items()}, destination / 'model.safetensors')
