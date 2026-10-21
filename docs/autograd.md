# Automatic differentiation

This guide describes the source implementation, not the published `0.9.9-alpha` release.

DeepJ records differentiable tensor operations when at least one input requires gradients.
Calling `backward()` follows those operations in reverse and accumulates gradients on leaf tensors.
The engine is Java code and uses the selected CPU or Metal backend, including fused backward kernels.

```java
import io.github.kirstenali.deepj.tensor.Tensor;

Tensor x = Tensor.from2D(new float[][]{{3.0f}}).requiresGrad(true);
Tensor loss = x.multiply(x).add(x).sumTensor();
loss.backward();
float derivative = x.grad().item(); // 7.0
```

For non-scalar outputs, pass an upstream tensor of the same shape to `output.backward(upstream)`.
Gradients accumulate until `tensor.zeroGrad()` or `model.zeroGrad()` is called.
A backward graph is released after use. Run forward again for another backward pass.
Higher-order derivatives and retaining graphs for repeated backward passes are not supported.
Graph connections retain shape and gradient metadata, not complete activation arrays. Each operation
saves only the numerical values its derivative needs, and releases them after backward.

## Layers and training

A layer implements `forward(...)` and supplies its parameters. It no longer implements `backward(...)`.
Repeated uses of one layer and shared parameters are handled by the operation graph, not mutable
last-forward caches. Each parameter's `value` is a tracked leaf tensor; `parameter.grad()` and
`parameter.value.grad()` access the same gradient buffer.

```java
model.zeroGrad();
Tensor prediction = model.forward(input);
Tensor loss = new MSELoss().forward(prediction, target);
loss.backward();
optimizer.step(model.parameters());
```

The existing trainers perform these steps automatically, including causal cross entropy,
response-only masks, gradient accumulation, clipping and AdamW updates. Differentiable losses return
tensors through `forward(...)`. `item()`, `sum()` and other methods returning Java numbers extract
values for logging or evaluation; use `sumTensor()` when a total must remain differentiable.

## Inference and updates

Use a thread-local, nestable scope when calling model forward directly for inference:

```java
import io.github.kirstenali.deepj.autograd.GradMode;

try (var ignored = GradMode.noGrad()) {
    Tensor logits = model.forward(tokens);
}
```

`TextGenerator`, the chat API, evaluation, initialization and AdamW already use the appropriate
no-grad paths. To freeze a parameter, call `parameter.value.requiresGrad(false)` between passes.
To update values manually, use `set(...)`, `setRow(...)` or in-place operations inside `noGrad()`.
Changing a tensor needed by an outstanding backward pass causes an error rather than using stale values.

`detach()` returns an independent, untracked copy. `copy()` preserves gradient tracking.
Neither introduces shared GPU-buffer ownership.

The public `data` array and `Tensor.backend()` are low-level escape hatches for kernels and advanced
code. Direct array writes and raw backend calls bypass autograd's managed mutation checks and tracking.
Do not use them to modify tensors belonging to a live backward graph. Use tensor operations for
differentiable code, and managed writes or `TensorAdapters.unpackF32Into(...)` for explicit value replacement.

## Migrating from manual backpropagation

- Replace `model.backward(gradient)` with `output.backward(gradient)` on the actual forward result.
- Remove layer/activation backward methods and last-forward caches.
- Use `LossFunction.forward(...)` for a differentiable scalar loss, then call `loss.backward()`.
- Replace the `Parameter.grad` field with `parameter.grad()`. Advanced callers can supply a detached
  gradient using `parameter.setGradient(...)`.
- Parameter tensor identity is fixed. Replace its values through managed writes rather than assigning
  another tensor to `parameter.value`.
- Backend derivative kernels remain available. They are implementation building blocks, not a second
  layer-by-layer training system.
- Tensor convenience wrappers for `reluBackward`, `geluBackward`, `softmaxBackward`,
  `crossEntropyGradient` and `layerNormBackward` are removed; autograd invokes the backend rules.

No model or training-checkpoint format changes are required. Saved parameter ordering, tied-weight
identity, optimizer moments and resume counters are unchanged. Runtime autograd graphs are not saved.

## Verification

The Maven build enforces the normal source/test quality rules and an autograd-specific minimum of
95% line and 90% branch coverage. Tests cover numerical derivatives, old manual-training fixtures,
CPU/Metal agreement, shared weights, masks, graph lifetime, mutation errors and checkpoint resumption.
Fixed-heap subprocess tests also check that shape-only chains and softmax graphs do not retain unused
activation arrays. Custom operations must return a new, untracked tensor rather than overwrite a graph.

The optional GitHub Actions benchmark compares synthetic 90M Prism and 100M Nova training against
a chosen baseline, at sequence length 1,024 and accumulation sizes one and two. It uses separate JVMs,
four warm-up updates and six measured updates per case, and records process peak memory. It reads no
training data or checkpoints. Autograd is not assumed to improve training speed or model quality.
