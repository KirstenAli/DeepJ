"""Optional reference: SNAPSHOT DTYPE INPUT_JSON [THREADS=10 WARMUPS=3 RUNS=3]."""
import json
import sys
from pathlib import Path
import torch
import transformers
from safetensors import safe_open
import qwen3_benchmark as reference


def load_float32_model(root):
    config = reference.Qwen3Config.from_pretrained(root, local_files_only=True)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = reference.Qwen3ForCausalLM(config)
    restore_float32_weights(model, root)
    model.tie_weights()
    model.requires_grad_(False).eval()
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config, device="cpu")
    return model.to("cpu")


def restore_float32_weights(model, root):
    index = root / "model.safetensors.index.json"
    names = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for name in sorted(names):
        with safe_open(root / name, framework="pt", device="cpu") as source:
            keys = list(source.keys())
        for key in keys:
            with safe_open(root / name, framework="pt", device="cpu") as source:
                value = source.get_tensor(key).to(dtype=torch.float32)
            model.load_state_dict({key: value}, strict=False, assign=True)


def main():
    root, name, path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    dtype = {"FLOAT32": torch.float32, "FLOAT16": torch.float16, "BFLOAT16": torch.bfloat16}[name]
    threads, warmups, runs = [int(sys.argv[i]) if len(sys.argv) > i else default
                             for i, default in [(4, 10), (5, 3), (6, 3)]]
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = load_float32_model(root) if dtype == torch.float32 else reference.load_model(root, "cpu", dtype=dtype)
    assert next(model.parameters()).dtype == dtype
    data = reference.input_tokens(root, path)
    print(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__,
                          threads=threads, interop=torch.get_num_interop_threads(), dtype=name)), flush=True)
    for number in range(-warmups, runs):
        result = reference.trial(model, data, "cpu", number)
        result["dtype"] = name
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
