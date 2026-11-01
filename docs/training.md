# Training neural networks

DeepJ networks train through the same tensor operations, autograd engine, parameter
objects, and optimizers. MLPs, CNNs, recurrent networks, autoencoders, and Transformer
encoders use this shared training path.

See [availability](README.md#availability) before using the `nn` package with a published
artifact. This guide describes the current-source APIs.

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

New `Module` instances start in training mode. `train()`, `eval()`, and
`setTraining(boolean)` propagate through new-module children. Dropout is active in
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

## Saving state and language-model training

[Network weight archives](persistence.md) store parameters and persistent buffers for
an already constructed network. Architecture, optimizer state, training counters,
recurrent state, and dropout RNG state are managed separately.

Causal language-model training has its own dataset, masking, clipping, evaluation,
and resumable-checkpoint utilities. See [language models](language-models.md) for
those APIs rather than assuming they apply to every network family.
