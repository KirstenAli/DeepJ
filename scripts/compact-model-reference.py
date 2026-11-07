"""Regenerate compact Gemma/SANA component oracles from the existing tiny FP32 checkpoints."""
import json
from pathlib import Path
import torch
import diffusers
import transformers
from safetensors.torch import load_file, save_file
from diffusers import AutoencoderDC, SanaTransformer2DModel, SanaVideoTransformer3DModel
from transformers import Gemma2Config, Gemma2Model

ROOT = Path(__file__).resolve().parents[1] / 'src/test/resources'
torch.set_num_threads(1)


def model(kind):
    folder = ROOT / ('sana-video' if kind == 'video' else 'sana')
    stage = folder / {'gemma': 'gemma', 'decoder': 'vae'}.get(kind, 'transformer')
    if kind == 'gemma':
        settings = json.loads((stage / "config.json").read_text())
        for name in ("attn_logit_softcapping", "final_logit_softcapping"):
            if settings.get(name) is not None:
                settings[name] = float(settings[name])
        config = Gemma2Config.from_dict(settings)
        config._attn_implementation = 'eager'
        value = Gemma2Model(config)
    else:
        cls = {'sana': SanaTransformer2DModel, 'video': SanaVideoTransformer3DModel, 'decoder': AutoencoderDC}[kind]
        value = cls.from_config(json.loads((stage / 'config.json').read_text()))
    value.load_state_dict(load_file(stage / 'model.safetensors'))
    return value.eval(), load_file(folder / 'model.safetensors')


def run(kind, value, fixtures, dtype):
    if kind == 'decoder':
        return value.decode(fixtures['vae2.input'].to(dtype)).sample
    if kind == 'gemma':
        return value(fixtures['gemma.tokens'].long(),
                     attention_mask=fixtures['gemma.mask'].long()).last_hidden_state
    inputs = dict(hidden_states=fixtures['transformer.input'].to(dtype),
                  encoder_hidden_states=fixtures['transformer.text'].to(dtype),
                  encoder_attention_mask=fixtures['transformer.mask'],
                  timestep=torch.tensor([733] if kind == 'video' else [0.6]))
    if kind == 'sana':
        inputs['guidance'] = torch.tensor([0.45])
    return value(**inputs).sample


def main():
    outputs = {}
    with torch.inference_mode():
        for kind in ('gemma', 'sana', 'video', 'decoder'):
            for name, dtype in (('FLOAT16', torch.float16), ('BFLOAT16', torch.bfloat16)):
                value, fixtures = model(kind)
                result = run(kind, value.to(dtype), fixtures, dtype).float().contiguous()
                if not torch.isfinite(result).all():
                    raise ValueError(f'Non-finite {kind} {name} reference')
                outputs[f'{kind}.{name}'] = result
    folder = ROOT / 'compact-models'
    folder.mkdir(exist_ok=True)
    save_file(outputs, folder / 'model.safetensors')
    (folder / 'reference.json').write_text(json.dumps(dict(torch=torch.__version__,
        transformers=transformers.__version__, diffusers=diffusers.__version__, device='cpu'), indent=2) + '\n')


if __name__ == '__main__':
    main()
