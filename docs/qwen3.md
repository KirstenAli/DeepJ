# Qwen3

DeepJ implements the dense Qwen3 decoder used by Qwen3-1.7B. It combines grouped-query causal attention,
per-head query/key RMS normalization, half-split rotary positions, pre-normalized residual blocks,
SiLU-gated feed-forward layers, final RMS normalization, and tied input/output embeddings.

All network arithmetic and differentiation use DeepJ's existing tensor operations and selected CPU or Metal backend.
The Qwen3 classes supply the model composition, checkpoint layout, and tokenization rules.

## Model and training

`io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3` implements `CausalLM`.
Its `forward(int[])` returns `[time, vocabulary]` logits; `forwardBatch(int[][])` returns
`[batch, time, vocabulary]` logits for rectangular token batches. Token IDs and sequence lengths are validated.

`Qwen3Config.onePointSevenB()` specifies the released model:

| Setting | Value |
| --- | ---: |
| Parameters, counting the tied embedding once | 1,720,574,976 |
| Decoder blocks | 28 |
| Hidden width | 2,048 |
| Feed-forward width | 6,144 |
| Query heads / key-value heads | 16 / 8 |
| Head width | 128 |
| Vocabulary rows | 151,936 |
| Context length | 40,960 |
| Rotary base | 1,000,000 |
| RMS normalization epsilon | 0.000001 |

Constructors accept a configuration and random seed, with an optional parameter storage `DType`.
The model works with the existing `CausalLMTraining`, differentiable losses, gradient accumulation, optimizers,
and data-parallel training APIs. Parameters require gradients; their gradient buffers are allocated when training
needs them. FLOAT32 gradients and accumulation are retained with FLOAT16/BFLOAT16 parameter storage.

The preset holds about 3.44 GB of parameters in BFLOAT16/FLOAT16, or 6.88 GB in FLOAT32.
Training additionally allocates gradients, activations, and optimizer state. Loading uses bounded per-tensor staging.

## Pretrained weights and persistence

`Qwen3Pretrained.load(Path)` reads a local Hugging Face snapshot containing `config.json` and either
`model.safetensors` or `model.safetensors.index.json` plus its shards. It preserves source storage dtypes.
The overload accepting `DType` converts every parameter to that format.
`Qwen3Pretrained.load(directory, dtype, true)` also keeps embedding activations in the
stored dtype. With all parameters in FLOAT16 or BFLOAT16, the model uses compact
activations through the shared tensor operations. The existing one- and two-argument
overloads retain their FLOAT32 embedding outputs. For new training models, pass
`ParameterFactory.uniform(dtype)` to the factory-accepting `Qwen3` constructor;
gradients and optimizer state remain FLOAT32.

The importer validates names, shapes, byte ranges, shard membership, and supported configuration settings before
constructing the model. It transposes projection matrices into DeepJ's layout and retains one shared embedding
parameter. If a separate `lm_head.weight` is present, it must be identical to the token embedding.

`Qwen3Checkpoint.save(Qwen3, Path)` writes the configuration and weights into a new directory in the same
Hugging Face format. The destination must not exist. Publication uses an atomic directory move after both files
have been written. `Qwen3Checkpoint.load(Path)` reconstructs the saved model. Tokenizer assets and optimizer state
are managed separately. Loading and saving do not reset the backend or invalidate other live models.
To restore compact activations from a saved Qwen snapshot, use
`Qwen3Pretrained.load(directory, null, true)`; the Hugging Face weight files preserve
parameter dtypes but do not store DeepJ's activation policy.

These entry points are independent of the existing model format registry.
The implementation targets the released dense, tied-embedding Qwen3 configuration with default RoPE and full causal
attention. Unsupported configuration options fail explicitly during loading.

## CLI and desktop UI

The normal `deepj chat`, `deepj generate`, and `deepj inspect` commands accept a Qwen3 snapshot directory,
its `config.json`, or a safetensors weight/index file inside that directory. Existing backend and sampling flags
apply. Inspection reads configuration and tensor metadata without constructing the model or selecting a backend.

The recommended Metal command explicitly selects BF16 weights and activations:

```bash
mvn package -Pcli
java -Xmx3g --enable-native-access=ALL-UNNAMED -jar target/deepj-cli.jar \
  --backend metal chat ./qwen3-1.7b --dtype bfloat16
```

The option selects the in-memory format at load time. Explicit `float16` and
`bfloat16` selections use compact activations and weights through the shared
CPU/Metal kernels. `float32` remains available when FP32 execution is wanted.
Omitting the dtype preserves checkpoint storage and the existing activation policy.

For desktop BF16 chat, Thinking and Agent mode, run:

```bash
JAVA_TOOL_OPTIONS="-Xmx3g -Ddeepj.chat.qwen3.dtype=bfloat16" mvn compile javafx:run
```

Use `-Ddeepj.chat.qwen3.dtype=float16` to select FP16. The desktop and CLI default
to a maximum of 2,048 generated tokens; generation also respects the model's
context window and end-of-sequence tokens.

In the desktop chat app, choose `config.json` or a safetensors file through the model picker. Keep the configuration,
tokenizer, index, and shards together in the snapshot directory. Thinking is off by default in both interfaces.
In the desktop app, open **Chat options** beside the message field and enable **Thinking** to use Qwen3's thinking
template, with or without Agent mode. The expandable **Thought process** appears above the answer in both modes.
Agent mode includes reasoning across tool calls in the same panel; its tool approval flow is unchanged.
Reasoning uses the same maximum token budget as the answer;
increase **Settings → Generation settings → Maximum tokens** for longer reasoning. An unfinished reasoning
response is labelled explicitly. Both interfaces recognize the model's end-of-sequence IDs. Tokenizer IDs may occupy
fewer rows than the model's padded vocabulary. Existing DeepJ checkpoint loading remains available.

## Tokenization and conversations

`io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Tokenizer.load(Path)` reads `tokenizer.json` from the local snapshot
and implements DeepJ's `Tokenizer` interface. It follows the file's Unicode normalization and pre-tokenization
regex, byte-level BPE merges, and added-token definitions. Encoding adds no BOS token, padding, or truncation.
Decoding supports optional removal of special tokens. Both released end-of-sequence IDs are recognized.

`Qwen3Chat.format(...)` formats text conversations with system, user, and assistant messages, optional assistant
reasoning, and thinking/non-thinking generation prompts. Previous-turn reasoning is removed according to the
released chat template. This typed formatter covers text messages; tool schemas, tool-call records, and multimodal
message objects are outside its input contract.

`openGeneration()` supplies the existing generation-session contract with a KV cache. Sessions support batched
prompt prefill, single-token continuation, and prompt resets. The `ContextWindowMode.ROLLING` overload retains a
rolling cache with absolute rotary positions; the default rebuilds at the context boundary. Close sessions when
finished. Cache capacity starts small and grows by copying existing keys and values; appended tokens do not
recompute the previous prefix when capacity increases. Changed prompts and context-boundary rebuilds reset the cache.
Updating weights or switching backends requires a new session. Qwen3 uses the same tensor-scope and
single-threaded model/session ownership conventions as the existing decoder implementations.

## Verification

The tests include an independent double-precision decoder, finite differences, every-parameter comparisons against
Transformers, checkpoint round trips, sharded imports, malformed-file checks, Unicode tokenization, chat-template
comparisons, cached decoding, and training with the existing optimizer. The numerical fixtures cover all three
parameter storage dtypes with FLOAT32 arithmetic in the reference model.

`Qwen3MetalTest` exercises physical Metal execution. Set `deepj.qwen3.requireMetal=true` to fail when no Metal device
is available. The optional full-vocabulary tokenizer comparison uses `deepj.qwen3.tokenizer` as its local snapshot
directory. The full 1.7B Metal comparison uses `deepj.qwen3.pretrained` for the snapshot and `deepj.qwen3.reference`
for independently generated Transformers logits; it requires sufficient Java heap and device memory.

`src/test/python/qwen3_pretrained_reference.py` generates the full-model reference JSON from a local snapshot.
`src/test/python/qwen3_benchmark.py` and the optional `Qwen3Benchmark` test-source entry point measure synchronized
inference using matching token IDs, FLOAT32 weights/activations, one warm-up trial, and three measured trials.
They report prompt prefill time and cached decode throughput separately, with model loading excluded.
The default probe uses one 64-token prompt and 16 fixed continuation tokens, materializing the full final logit row
after each forward pass. Set matching thread counts when comparing CPU results. PyTorch uses its SDPA attention path.
These probes measure inference rather than training; the Java probe needs sufficient heap for FLOAT32 weights and
the existing CPU projection's temporary transpose.

Reference sources: [official model and weights](https://huggingface.co/Qwen/Qwen3-1.7B),
[released configuration](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json),
and [Transformers 4.51.3 Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py).
The model and reference implementation use the Apache 2.0 license.
