"""Independent Gemma 2 chat oracles: PyTorch 2.8.0 and Transformers 4.51.3.

Only regenerates small test fixtures. DeepJ inference does not invoke Python.
"""
from pathlib import Path
import json
import string
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, normalizers, decoders, processors, AddedToken
from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1] / "src/test/resources/gemma2"
ROOT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(742)
torch.set_num_threads(1)
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
for char in string.printable.replace(" ", "") + "▁":
    if char not in vocab:
        vocab[char] = len(vocab)
while len(vocab) < 106:
    vocab[f"<unused_{len(vocab)}>"] = len(vocab)
vocab.update({"<start_of_turn>": 106, "<end_of_turn>": 107})
for value in range(256):
    vocab[f"<0x{value:02X}>"] = len(vocab)
while len(vocab) < 384:
    vocab[f"<unused_{len(vocab)}>"] = len(vocab)
backend = Tokenizer(models.BPE(vocab=vocab, merges=[], byte_fallback=True, unk_token="<unk>"))
backend.normalizer = normalizers.Replace(" ", "▁")
backend.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Replace("▁", " "), decoders.Fuse()])
backend.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
                                   pad_token="<pad>", unk_token="<unk>")
tokenizer.add_special_tokens({"additional_special_tokens": [
    AddedToken("<start_of_turn>", normalized=False), AddedToken("<end_of_turn>", normalized=False)]})
tokenizer.chat_template = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<start_of_turn>' + ('model' if message['role'] == 'assistant' else 'user') + '\\n'"
    " + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}")
tokenizer.save_pretrained(ROOT)

config = Gemma2Config(vocab_size=384, hidden_size=18, intermediate_size=24, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=4, max_position_embeddings=512,
                      sliding_window=4, query_pre_attn_scalar=4, attn_logit_softcapping=50,
                      final_logit_softcapping=30, hidden_activation="gelu_pytorch_tanh",
                      attn_implementation="eager")
model = Gemma2ForCausalLM(config).eval()
model.save_pretrained(ROOT)
ids = torch.tensor([[2, 18, 21, 13, 24, 17, 10]])
reference = {}
with torch.no_grad():
    reference["tokens"] = ids.float()
    reference["logits"] = model(ids, use_cache=False).logits.contiguous()
    long_ids = torch.tensor([[2] + [10 + index % 80 for index in range(128)]])
    reference["long_tokens"] = long_ids.float()
    reference["long_logits"] = model(long_ids, use_cache=False).logits[:, -1].contiguous()
    output = ids.clone()
    generated = []
    for step in range(4):
        token = model(output, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        generated.append(int(token.item()))
        output = torch.cat([output, token], dim=-1)
    reference["greedy"] = torch.tensor(generated, dtype=torch.float32)
    chat_ids = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}],
                                             add_generation_prompt=True, return_tensors="pt")
    output = chat_ids.clone()
    for step in range(4):
        token = model(output, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        if int(token.item()) in (1, 107):
            break
        output = torch.cat([output, token], dim=-1)
    chat_reply = tokenizer.decode(output[0, chat_ids.shape[1]:])

loss = model(ids, use_cache=False).logits.square().mean()
loss.backward()
reference["embedding_gradient"] = model.model.embed_tokens.weight.grad.contiguous()
model.zero_grad(set_to_none=True)
with torch.no_grad():
    reference["logits_bf16"] = model.to(torch.bfloat16)(ids, use_cache=False).logits.float().contiguous()
(ROOT / "oracles").mkdir(exist_ok=True)
save_file(reference, ROOT / "oracles/model.safetensors")

messages = [{"role": "user", "content": " Hello "}, {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "A fox 🦊\n雪"}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
cases = [{"text": text, "ids": tokenizer.encode(text)} for text in ["Hello", "A fox 🦊\n雪"]]
cases.append({"text": formatted.removeprefix("<bos>"), "ids": tokenizer.encode(formatted, add_special_tokens=False)})
(ROOT / "tokenizer-reference.json").write_text(json.dumps(
    {"cases": cases, "formatted": formatted, "chat_reply": chat_reply}, indent=2) + "\n")
