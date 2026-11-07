"""Summarize the fresh-process timings and untimed numerical checks from run.sh."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(left, right):
    first, second = load_file(str(left)), load_file(str(right))
    assert first.keys() == second.keys()
    assert all(first[key].shape == second[key].shape for key in first)
    a = np.concatenate([first[key].flatten().astype(np.float64) for key in sorted(first)])
    b = np.concatenate([second[key].flatten().astype(np.float64) for key in sorted(second)])
    assert np.isfinite(a).all() and np.isfinite(b).all()
    error = a - b
    return dict(max_abs=float(np.max(np.abs(error))), rms=float(np.sqrt(np.mean(error ** 2))),
                relative_l2=float(np.linalg.norm(error) / np.linalg.norm(b)),
                deepj_l2=float(np.linalg.norm(a)), pytorch_l2=float(np.linalg.norm(b)))


def audit(root, backend, reference):
    java, torch = root / f"java-{backend}-audit", root / f"torch-{reference}-audit"
    java_raw, torch_raw = java.with_name(java.name + "-raw"), torch.with_name(torch.name + "-raw")
    result = {name: compare(java_raw / file, torch_raw / file) for name, file in
              (("logits", "logits.safetensors"), ("raw_gradients", "gradients.safetensors"))}
    result["after_one_update"] = compare(java / "model.safetensors", torch / "model.safetensors")
    result["deepj"] = read(java_raw / "result.json")
    result["pytorch"] = read(torch_raw / "result.json")
    assert result["logits"]["max_abs"] < 1e-5
    assert result["raw_gradients"]["relative_l2"] < 1e-4
    assert result["after_one_update"]["relative_l2"] < 1e-5
    return result


def runs(root, name):
    results = [read(root / f"{name}-{index}" / "result.json") for index in (1, 2)]
    for result in results:
        assert result["samples"] == 30 and result["warmups"] == 100
        assert len(result["ms"]) == 30 and len(result["losses"]) == 130
        assert np.isfinite(result["ms"]).all() and np.isfinite(result["losses"]).all()
        result["median_ms"] = statistics.median(result["ms"])
    return dict(median_ms=statistics.median(run["median_ms"] for run in results), runs=results)


def final_comparison(root, backend, reference):
    java, torch = root / f"java-{backend}-1", root / f"torch-{reference}-1"
    losses_a = np.array(read(java / "result.json")["losses"])
    losses_b = np.array(read(torch / "result.json")["losses"])
    return dict(weights=compare(java / "model.safetensors", torch / "model.safetensors"),
                max_loss_difference=float(np.max(np.abs(losses_a - losses_b))),
                deepj_final_loss=float(losses_a[-1]), pytorch_final_loss=float(losses_b[-1]))


def summarize(root, jar):
    result = dict(config=read(root / "fixture/config.json"), input=read(root / "fixture/input.json"))
    result["sha256"] = dict(jar=digest(jar), initial_weights=digest(root / "fixture/model.safetensors"))
    result["runs"] = {name: runs(root, name) for name in ("java-cpu", "torch-cpu", "java-metal", "torch-mps")}
    result["audit"] = {backend: audit(root, backend, ref) for backend, ref in (("cpu", "cpu"), ("metal", "mps"))}
    result["after_130_updates"] = {backend: final_comparison(root, backend, ref)
                                    for backend, ref in (("cpu", "cpu"), ("metal", "mps"))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("jar", type=Path)
    args = parser.parse_args()
    result = summarize(args.output, args.jar)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value["median_ms"] for key, value in result["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
