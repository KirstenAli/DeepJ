# Training a model

DeepJ networks train through the same tensor operations, autograd engine, parameter
objects, and optimizers. This guide covers feed-forward networks, CNNs, recurrent
networks, autoencoders, Transformer encoders, and causal language models.

This guide describes the DeepJ `0.13.0-alpha` training APIs. See
[version compatibility](README.md#availability) when using an older dependency.

## Choose a training API

| Model type or objective | Training entry point |
|---|---|
| `FNN`, CNNs, and other `Layer` networks with tensor inputs and targets | `io.github.kirstenali.deepj.nn.NdSupervisedTraining.trainer(...)` for first-axis minibatches |
| Autoencoders | `NdSupervisedTraining.trainer(...)` with a reconstruction loss such as `MSELoss`; use the input as the target |
| RNN, GRU, LSTM, or Transformer encoder with an ordinary `forward(input)` objective | `NdSupervisedTraining.trainer(...)`, preserving each sample's sequence axes |
| Recurrent state, padding masks, multiple inputs, or a custom loss | `new Trainer(stepFunction)` with explicit model calls, backward, and optimizer updates |
| Causal decoder models, including Qwen3 | `io.github.kirstenali.deepj.training.CausalLMTraining.trainer(...)` with token batches and optional response masks |

For a single machine, select the [CPU or Metal backend](backends.md) before
constructing tensors and models. Batching, forward operations, and gradients use
that backend. For multiple workers, follow the [distributed training guide](../distributed/README.md).

## Training cycle

A differentiable training step follows these operations:

1. Clear the model's parameter gradients with `model.zeroGrad()`.
2. Call `model.forward(input)` to obtain a prediction tensor.
3. Call a loss function's `forward(prediction, target)` to obtain a scalar tensor.
4. Call `backward()` on the loss tensor.
5. Pass `model.parameters()` to the optimizer's `step(...)` method.

`Layer` and `Trainable` expose the parameter contract. `Parameter.value` is a tracked
leaf tensor, and its gradient is available through `Parameter.grad()`. Shared parameter
tensors in `Sequential` are listed once. See [autograd](autograd.md) for accumulation,
graph lifetime, mutation checks, and FLOAT32 gradient storage.

## Batches and trainers

`NdSupervisedTraining.trainer(model, loss, optimizer, inputs, targets, seed)` creates
an existing `Trainer` with first-axis minibatching. Inputs and targets must have the
same positive sample count on axis 0. Remaining axes belong to each sample, so image
and sequence layouts remain intact.

A batch size smaller than the sample count samples indices with replacement using
the supplied seed; otherwise the trainer uses the complete input and target tensors.
Selection uses backend `indexSelect` operations. The existing matrix-only
`SupervisedTraining` API remains available for rank-two datasets.

`Trainer.trainStep(batchSize)` executes one update. `Trainer.train(...)` supports
step limits, loss logging, exponential moving averages, optional loss-based stopping,
and step hooks. Those controls are step-based, not an automatic epoch-based data pipeline.
Use a custom `Trainer.StepFunction` when the training step needs extra inputs or controls.

## Losses and optimizers

`MSELoss` computes mean squared error over equal, nonempty prediction and target shapes.
It is suitable for regression and reconstruction objectives, including autoencoders.
`CrossEntropyLoss` accepts logits and integer class targets for classification objectives.
Use a loss's tensor-returning `forward(...)` method when gradients are required.

`AdamW` updates the listed parameters, with configurable learning rate, moments,
epsilon, and weight decay. Existing scheduling and gradient-clipping utilities can
be used where the chosen training path exposes them. The generic ND supervised
trainer performs batching, backward, and optimizer updates; it does not automatically
apply a learning-rate schedule or gradient clipping.

## Training and evaluation mode

`Module` instances start in training mode. `train()`, `eval()`, and
`setTraining(boolean)` propagate through child modules. Dropout is active in
training and disabled during evaluation. BatchNorm uses batch statistics during
training and persistent running statistics during evaluation.

Mode selection and gradient recording are independent. `eval()` does not disable
autograd, and `GradMode.noGrad()` does not disable dropout or BatchNorm updates.
For evaluation, choose the intended mode and gradient scope explicitly. Use tensor
scopes for temporary evaluation outputs; see [CPU and Metal](backends.md).

## Recurrent and masked inputs

Recurrent layers consume `[batch, time, features]` tensors. `Recurrent.run(...)`
accepts an explicit initial state and optional per-sample sequence lengths. State can
be passed between chunks, or detached for truncated backpropagation. Padding behavior
and zero-length samples are defined in the [network guide](neural-networks.md#recurrent-state-and-masks).

Transformer encoder masks identify valid tokens for attention and pooling. The generic
ND trainer calls `forward(input)`; training that needs explicit masks, recurrent state,
or custom sequence losses should use a custom step with those APIs.

## Causal language-model training

`CausalLMTraining.trainer(model, dataset, optimizer)` trains a `CausalLM` from a
`BatchSource`. Each `Batch` contains token inputs, next-token targets, and optional
per-sequence loss masks. A response mask selects the tokens that contribute to
the loss, such as the assistant response during instruction fine-tuning.

The trainer computes cross entropy, averages sequence losses and gradients over
the batch, clips the global gradient norm using `model.gradClipNorm()`, and updates
the parameters. It uses batched forward when the model supports it. The overload
accepting a learning rate creates an AdamW optimizer.

`GradientAccumulation.trainer(model, dataset, optimizer, steps)` combines several
microbatches into one optimizer update. Calling `trainStep(microBatchSize)` produces
an effective batch of `microBatchSize * steps` sequences.

See [language-model data and persistence](language-models.md#text-data-and-persistence),
[looped decoder training](looped-training.md), and [Qwen3 model and training](qwen3.md#model-and-training)
for the corresponding model and dataset contracts.

## Distributed training

The optional [distributed module](../distributed/README.md) lets multiple machines
train a model together using synchronous data parallelism. It supports CPU workers
or Apple Silicon Macs using Metal. The model type determines the architecture;
distributed training determines how the training work is shared between workers.
Run one JVM per worker, with the same model architecture and optimizer settings,
and give each worker a different data shard. Every worker holds a full model
replica; gradients are averaged across workers before the optimizer update. See
[how it works](../distributed/README.md#how-it-works) for the update sequence.

Use `TensorObjective` for ordinary `Layer` networks, `CausalObjective` for causal
language models, or `TrainingObjective` for a custom differentiable objective.
`DistributedTrainer` manages initialization, weighted gradient synchronization,
clipping, and updates; `trainer()` exposes the usual `Trainer` interface.

The [worker setup](../distributed/README.md#workers-and-synchronization),
[data sharding](../distributed/README.md#losses-gradients-and-data), and
[distributed checkpoints](../distributed/README.md#checkpoints) sections cover TCP
connections, rank assignment, uneven batches, and continuation on every worker.
Build the module separately using its [build instructions](../distributed/README.md#build-and-verification).

## Saving training state

[Network weight archives](persistence.md) store parameters and persistent buffers for
an already constructed network. Architecture, optimizer state, training counters,
recurrent state, and dropout RNG state are managed separately.

Causal language models have [resumable training checkpoints](persistence.md),
while distributed jobs use `DistributedCheckpoint` on every rank. Restore the
matching model, optimizer, and data-source state when continuing a training run.
