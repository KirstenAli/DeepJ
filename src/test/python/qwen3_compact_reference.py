"""Generate compact-activation logits from the existing official Qwen3 fixtures."""
import json
from pathlib import Path
import torch
import transformers
from transformers import Qwen3ForCausalLM


@torch.inference_mode()
def generate(directory, dtype):
    model = Qwen3ForCausalLM.from_pretrained(directory, torch_dtype=dtype,
                                            attn_implementation="eager").eval()
    previous = json.loads((directory / "expected.json").read_text())
    tokens = torch.tensor(previous["tokens"])
    logits = model(tokens, use_cache=False).logits.float().flatten().tolist()
    result = dict(tokens=previous["tokens"], logits=logits, dtype=str(dtype),
                  torch=torch.__version__, transformers=transformers.__version__)
    (directory / "compact.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "resources" / "qwen3"
    for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
        generate(root / name, dtype)
