"""Optional PyTorch inference benchmark: SNAPSHOT DEVICE INPUT_JSON [THREADS]."""
import json
import sys
import time
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def load_model(root, device, attention="sdpa", dtype=torch.float32):
    config = Qwen3Config.from_pretrained(root, local_files_only=True)
    config._attn_implementation = attention
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    restore_shards(model, root)
    model.requires_grad_(False).to(dtype=dtype).eval()
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config, device="cpu")
    model.to(device)
    model.tie_weights()
    return model


def restore_shards(model, root):
    index = root / "model.safetensors.index.json"
    names = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for name in sorted(names):
        model.load_state_dict(load_file(root / name), strict=False, assign=True)
    model.tie_weights()


def input_tokens(root, path):
    if not path.exists():
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        prompt = tokenizer.encode("Explain how neural networks learn from examples and use that knowledge. " * 10)[:64]
        decode = tokenizer.encode(" A fixed continuation measures the speed of cached token generation. " * 3)[:16]
        path.write_text(json.dumps(dict(prompt=prompt, decode=decode)))
    return json.loads(path.read_text())


@torch.inference_mode()
def forward(model, tokens, cache, device):
    result = model(torch.tensor([tokens], device=device), past_key_values=cache,
                   use_cache=True, logits_to_keep=1)
    logits = result.logits[0, -1].float().cpu().numpy().copy()
    return result.past_key_values, float(logits[0])


def trial(model, data, device, number):
    start = time.perf_counter()
    cache, checksum = forward(model, data["prompt"], None, device)
    prefill = time.perf_counter() - start
    start = time.perf_counter()
    for token in data["decode"]:
        cache, value = forward(model, [token], cache, device)
        checksum += value
    seconds = time.perf_counter() - start
    return dict(engine="PyTorch", backend=device, dtype="FLOAT32", trial=number,
                prefill_seconds=prefill, decode_seconds=seconds,
                tokens_per_second=len(data["decode"]) / seconds, checksum=checksum)


if __name__ == "__main__":
    snapshot, backend, input_file = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 10)
    torch.set_num_interop_threads(1)
    network = load_model(snapshot, backend)
    inputs = input_tokens(snapshot, input_file)
    for iteration in range(-1, 3):
        print(json.dumps(trial(network, inputs, backend, iteration)), flush=True)
