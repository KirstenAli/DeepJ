"""Optional matched FP32 training reference; requires torch 2.8.0 / transformers 4.51.3."""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("backend", choices=("cpu", "mps"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args()


def load_model(directory, device):
    config = Qwen3Config.from_pretrained(directory, local_files_only=True)
    config._attn_implementation = "sdpa"
    config.sliding_window = None
    model = Qwen3ForCausalLM(config)
    state = load_file(str(directory / "model.safetensors"))
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    return model.train().to(device)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def clip_gradients(parameters):
    gradients = [parameter.grad for parameter in parameters]
    norm = torch.nn.utils.get_total_norm(gradients, norm_type=2).item()
    if not math.isfinite(norm):
        raise ValueError("Non-finite gradient")
    if norm > 1.0:
        torch._foreach_mul_(gradients, 1.0 / (norm + 1e-12))


def train_step(model, optimizer, parameters, tokens, targets):
    optimizer.zero_grad(set_to_none=False)
    logits = model(tokens, use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    value = loss.item()
    if not math.isfinite(value):
        raise ValueError("Non-finite loss")
    clip_gradients(parameters)
    optimizer.step()
    return value


def save_state(model, output):
    parameters = dict(model.named_parameters())
    save_file({name: value.detach().cpu().contiguous() for name, value in parameters.items()},
              str(output / "model.safetensors"))


def audit(model, tokens, targets, output):
    raw = output.with_name(output.name + "-raw")
    raw.mkdir(parents=True, exist_ok=False)
    logits = model(tokens, use_cache=False).logits
    save_file({"logits": logits.squeeze(0).detach().cpu().contiguous()}, str(raw / "logits.safetensors"))
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    save_file({name: value.grad.detach().cpu().contiguous() for name, value in model.named_parameters()},
              str(raw / "gradients.safetensors"))
    norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters()]).item()
    (raw / "result.json").write_text(json.dumps(dict(loss=loss.item(), norm=norm)) + "\n")
    model.zero_grad(set_to_none=False)


def measure(model, tokens, targets, args):
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    warmups, samples = (0, 1) if args.audit else (100, 30)
    losses, times = [], []
    synchronize(args.backend)
    for iteration in range(-warmups, samples):
        start = time.perf_counter_ns()
        loss = train_step(model, optimizer, parameters, tokens, targets)
        synchronize(args.backend)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        losses.append(loss)
        if iteration >= 0:
            times.append(elapsed)
    return dict(engine="PyTorch", backend=args.backend, dtype="FLOAT32", warmups=warmups,
                samples=samples, ms=times, losses=losses, parameters=sum(p.numel() for p in parameters))


def prepare(args):
    torch.set_num_threads(10)
    torch.set_num_interop_threads(1)
    if args.backend == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Physical MPS required")
    args.output.mkdir(parents=True, exist_ok=False)
    model = load_model(args.fixture, args.backend)
    data = json.loads((args.fixture / "input.json").read_text())
    tokens = torch.tensor([data["tokens"]], device=args.backend)
    targets = torch.tensor([data["targets"]], device=args.backend)
    return model, tokens, targets


def main():
    args = arguments()
    model, tokens, targets = prepare(args)
    if args.audit:
        audit(model, tokens, targets, args.output)
    result = measure(model, tokens, targets, args)
    save_state(model, args.output)
    result.update(torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version())
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
