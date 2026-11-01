# Incremental generation

Model-aware `TextGenerator.generate(...)`, `TextGenerator.stream(...)`, `DeepJChat`, the desktop UI and the CLI use a separate key/value cache for each request. Origin, Orbit, Prism, Nova and Loop support this on CPU and Metal. Generic `Function<int[], Tensor>` overloads remain uncached.

The first forward pass processes the prompt. Subsequent passes process only the newly appended token, reusing earlier attention keys and values. Only the final hidden row is projected to vocabulary logits. This does not change the model's parameters, checkpoint format, training path, prompt format or sampling settings.

Nova and Loop keep their shared key/value heads compact during training and inference. CPU and fallback Metal attention batch query groups against the shared heads; the fused Metal kernels address those heads directly and accumulate their shared gradients without atomics. Prism caches its projected keys and values, not a compressed latent cache.

## Explicit sessions

For custom decoding, pass the complete current context on each call:

```java
try (GenerationSession session = model.openGeneration()) {
    Tensor promptLogits = session.apply(new int[]{1, 2, 3});
    // Consume/materialize the logits before the next call or temporary-buffer cleanup.
    Tensor nextLogits = session.apply(new int[]{1, 2, 3, 4});
}
```

Cached sessions return one logit row. Closing the session releases its cache retention, including after cancellation or failure. Separate sessions never reuse each other's cache. Loop has a distinct cache for every logical attention visit, including repeated visits to the same shared core layer.

## Default generation

The default `REBUILD` mode reuses the cache when the supplied context is the previous context plus one token. Other contexts rebuild the cache. Once the window is full, each new token drops the oldest token and shifts the window. Rebuilding preserves cropped-context generation: positions restart at zero, and the remaining tokens are recalculated without the dropped token. Their previous cached states were computed with that token present, even for models using RoPE.

This mode accelerates generation while the context grows, but loses incremental decoding once the window shifts on every step. Optional [rolling generation](#rolling-generation) avoids those repeated rebuilds for RoPE models, with different behaviour after eviction. Neither mode changes training.

## Rolling generation

Orbit, Prism, Nova and Loop can opt into a bounded rolling KV cache:

```java
DeepJChat chat = DeepJChat.load(Path.of("downloaded-model"))
        .withContextWindow(ContextWindowMode.ROLLING);
String answer = chat.generate("What is a computer?");
```

Import `io.github.kirstenali.deepj.nn.transformer.ContextWindowMode`. For explicit sessions, use
`model.openGeneration(ContextWindowMode.ROLLING)` with the same complete-context calling convention.

Until the first token is evicted, rolling mode follows the same cached calculation as the default
`REBUILD` mode. After that, it overwrites the oldest key/value entries in fixed-capacity storage,
processes only the new token and continues the RoPE positions. Position rows beyond the original
table are computed on demand; neither the KV cache nor the persistent position tables grow.

This changes the attention semantics after eviction. Retained cached states still contain influence
from earlier tokens, unlike a fresh calculation over only the latest window. Rolling mode is not
an exact shortcut for cropped-context recomputation, does not guarantee better answers and does not
extend the model's trained context capability. It changes neither weights nor training.

Each full-length context supplied after capacity is reached must represent exactly one token shift.
Even an identical window of repeated tokens counts as a new step. Start a new session to restart
the same prompt. Changed contexts without the required overlap reset the request's cache and positions.

In the desktop UI, select **Rolling KV cache** under Settings → Generation settings. In the CLI,
add `--rolling-cache`, or use `/set context-window rolling`. The default remains `REBUILD`.
Origin and custom models without rolling support report an error instead of silently changing
their positional behaviour. Long-output quality still needs to be evaluated for each checkpoint.

Changing model weights or the global tensor backend invalidates an open session; close it and create another. Custom `CausalLM` implementations keep the ordinary forward fallback. A `DecoderOnlyModel` extension must explicitly opt into cached generation and use layers that support incremental inference.

Floating-point matrix operations can round differently for a single token and a full prefix. Logit comparisons use numerical tolerances; identical sampling settings do not guarantee byte-identical text if a small logit difference changes a sampled choice. No repetition penalty, temperature or seed is changed by caching.
