"""Regenerate tiny numerical fixtures with transformers 4.51.3 and torch 2.8.0 (CPU)."""
import json
from pathlib import Path
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM


def model_config():
    return Qwen3Config(vocab_size=19, hidden_size=12, intermediate_size=20,
                       num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=4, max_position_embeddings=16,
                       rope_theta=1_000_000, tie_word_embeddings=True,
                       attention_dropout=0, rms_norm_eps=1e-6,
                       bos_token_id=None, eos_token_id=None, pad_token_id=None,
                       attn_implementation="eager")


def generate(root, dtype, name):
    torch.manual_seed(417)
    model = Qwen3ForCausalLM(model_config()).to(dtype).eval()
    destination = root / name
    model.save_pretrained(destination, safe_serialization=True)
    model.float()
    tokens = torch.tensor([[1, 5, 7, 3], [4, 6, 2, 8]])
    logits = model(tokens, use_cache=False).logits
    upstream = torch.linspace(-0.3, 0.4, logits.numel()).reshape(logits.shape)
    (logits * upstream).sum().backward()
    expected = dict(tokens=tokens.tolist(), logits=logits.detach().flatten().tolist(),
                    upstream=upstream.flatten().tolist(),
                    gradients={key: value.grad.flatten().tolist()
                               for key, value in model.named_parameters()},
                    torch=torch.__version__, transformers=transformers.__version__)
    (destination / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    (destination / "generation_config.json").unlink()


if __name__ == "__main__":
    fixture_root = Path(__file__).resolve().parents[1] / "resources" / "qwen3"
    for fixture_dtype, fixture_name in ((torch.float32, "float32"),
                                      (torch.float16, "float16"),
                                      (torch.bfloat16, "bfloat16")):
        generate(fixture_root, fixture_dtype, fixture_name)
