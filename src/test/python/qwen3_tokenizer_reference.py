"""Generate tokenizer/chat checks from a local, official Qwen3 tokenizer snapshot."""
import json
import random
import sys
from pathlib import Path
from transformers import AutoTokenizer


def texts():
    cases = ["", "Hello world!", "  leading\tand trailing  ", "1234567890",
             "I'm WE'LL you've I'd that's", "a\r\nb\n\n c", "cafe\u0301 e\u0301",
             "你好，世界！", "مرحبا بالعالم", "नमस्ते दुनिया", "👩🏽‍💻🧠🙂", "a\u00a0b\u2003c",
             "<|im_start|>user\nHello<|im_end|>\n<think>thought</think>", "a" * 500]
    random.seed(884)
    alphabet = " abcXYZ01289\r\n\t.,!'éñø中你好世界مرحباЖдनमस्ते🙂👩🏽‍💻\u00a0\u2003\u200b"
    cases += ["".join(random.choices(alphabet, k=random.randint(1, 120))) for _ in range(200)]
    return cases


def conversations():
    return [[dict(role="user", content="Hello")],
            [dict(role="system", content="Be concise."), dict(role="user", content="Why?")],
            [dict(role="user", content="One"), dict(role="assistant", content="<think>old</think>\nAnswer"),
             dict(role="user", content="Two")],
            [dict(role="user", content="Q"), dict(role="assistant", content="A", reasoning_content="\nReason\n")],
            [dict(role="user", content="Q"), dict(role="assistant", content="\nA")],
            [dict(role="assistant", content="A"), dict(role="system", content="System")]]


def generate(snapshot, output):
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded = [dict(text=text, ids=tokenizer.encode(text, add_special_tokens=False)) for text in texts()]
    for case in encoded:
        case["decoded"] = tokenizer.decode(case["ids"], skip_special_tokens=False)
        case["stripped"] = tokenizer.decode(case["ids"], skip_special_tokens=True)
    (output / "tokenizer-golden.json").write_text(json.dumps(dict(cases=encoded)), encoding="utf-8")
    chat = [dict(messages=messages, thinking=thinking, generation=generation,
                 expected=tokenizer.apply_chat_template(messages, tokenize=False,
                           add_generation_prompt=generation, enable_thinking=thinking))
            for messages in conversations() for thinking in (True, False) for generation in (True, False)]
    (output / "chat-golden.json").write_text(json.dumps(dict(cases=chat)), encoding="utf-8")


if __name__ == "__main__":
    generate(sys.argv[1], Path(__file__).resolve().parents[1] / "resources" / "qwen3")
