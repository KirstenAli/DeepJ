"""Small Gemma BF16 oracle: PyTorch 2.8.0, Transformers 5.22.1; no model download."""
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

ROOT = Path(__file__).resolve().parents[1] / "src/test/resources/gemma2/precision"
ROOT.mkdir(exist_ok=True)
torch.set_num_threads(1)
generator = torch.Generator().manual_seed(43)
query = (torch.randn(1, 2, 7, 8, generator=generator) * 3).bfloat16()
key = (torch.randn(1, 2, 7, 8, generator=generator) * 2).bfloat16()
value = torch.randn(1, 2, 7, 8, generator=generator).bfloat16()
mask = torch.full((1, 1, 7, 7), -1e30).triu(1)
config = Gemma2Config(
    hidden_size=16, intermediate_size=32,
    num_attention_heads=2, num_key_value_heads=2, head_dim=8,
)
positions = torch.arange(5, 12)[None]
rotary = Gemma2RotaryEmbedding(config)
cosine, sine = rotary(query, positions)
rotated, _ = apply_rotary_pos_emb(query, key, cosine, sine)
module = SimpleNamespace(num_key_value_groups=1, head_dim=8, training=False)
context, probabilities = eager_attention_forward(
    module, query, key, value, mask.bfloat16(), scaling=0.3, softcap=2.0,
)
values = {
    "query": query.float(), "key": key.float(), "value": value.float(),
    "mask": mask, "rotated": rotated.float(),
    "attention": context.transpose(1, 2).float(),
    "probabilities": probabilities.float(),
}
save_file({name: tensor.contiguous() for name, tensor in values.items()}, ROOT / "model.safetensors")
print("Saved independent BF16 rotary and softcap oracle", flush=True)
