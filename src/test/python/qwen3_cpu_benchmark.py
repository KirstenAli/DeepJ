"""Optional reference: SNAPSHOT DTYPE INPUT_JSON [THREADS=10 WARMUPS=3 RUNS=3]."""
import json
import sys
from pathlib import Path
import torch
import transformers
import qwen3_benchmark as reference


def main():
    root, name, path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    dtype = {"FLOAT32": torch.float32, "FLOAT16": torch.float16, "BFLOAT16": torch.bfloat16}[name]
    threads, warmups, runs = [int(sys.argv[i]) if len(sys.argv) > i else default
                             for i, default in [(4, 10), (5, 3), (6, 3)]]
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = reference.load_model(root, "cpu", dtype=dtype)
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
