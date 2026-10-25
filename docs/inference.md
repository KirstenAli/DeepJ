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

If the supplied context is not the previous context plus one token, the session re-prefills it. In particular, once a sliding window reaches its configured limit, generation re-prefills after each window shift. This preserves the previous position-reset behaviour rather than reusing incompatible cached states. Caching therefore does not accelerate decoding at a continually shifting context limit.

Changing model weights or the global tensor backend invalidates an open session; close it and create another. Custom `CausalLM` implementations keep the ordinary forward fallback. A `DecoderOnlyModel` extension must explicitly opt into cached generation and use layers that support incremental inference.

Floating-point matrix operations can round differently for a single token and a full prefix. Logit comparisons use numerical tolerances; identical sampling settings do not guarantee byte-identical text if a small logit difference changes a sampled choice. No repetition penalty, temperature or seed is changed by caching.
