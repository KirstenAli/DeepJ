"""Generate full Qwen3-1.7B FP32 reference logits from a local official snapshot: SNAPSHOT OUTPUT_JSON."""
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
from qwen3_benchmark import load_model
import torch


def reference(model, tokenizer, prompt):
    text = tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    cases = []
    for _ in range(3):
        with torch.inference_mode():
            logits = model(torch.tensor([tokens]), use_cache=False, logits_to_keep=1).logits[0, -1]
        next_token = int(logits.argmax())
        cases.append(dict(tokens=tokens.copy(), logits=logits.tolist(), next=next_token))
        tokens.append(next_token)
    return cases


if __name__ == "__main__":
    snapshot = Path(sys.argv[1])
    network = load_model(snapshot, "cpu", "eager")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    cases = [case for prompt in ("Reply with one word: hello", "What is 2 + 2?")
             for case in reference(network, tokenizer, prompt)]
    Path(sys.argv[2]).write_text(json.dumps(dict(cases=cases)))
