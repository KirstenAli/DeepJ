# Neural networks

The `io.github.kirstenali.deepj.nn` package provides neural-network implementations
and composable layers on DeepJ's shared `Layer`, `Parameter`, `Tensor`, autograd, and
optimizer APIs. See [availability](README.md#availability) for source/release details.

## Network implementations

| API | Input layout | Behavior |
|---|---|---|
| `MLP` | `[..., features]` | Delegates to the existing `FNN`; configurable hidden and output activations. |
| `Autoencoder` | Defined by its encoder | Composes encoder and decoder layers; exposes `encode` and `decode`. |
| `Conv1D` | `[batch, channels, length]` | Trainable cross-correlation with stride, symmetric zero padding, dilation and optional bias. |
| `Conv2D` | `[batch, channels, height, width]` | Spatial cross-correlation; `Window2D` configures each spatial dimension. |
| `Pool1D`, `Pool2D` | Channel-first sequences/images | Max or average pooling with configurable windows. |
| `GlobalAveragePool` | `[batch, channels, ...]` | Averages spatial axes, returning `[batch, channels]`. |
| `RNN`, `GRU`, `LSTM` | `[batch, time, features]` | Return every time step; explicit initial/final states and optional per-sample lengths. |
| `RNNCell`, `GRUCell`, `LSTMCell` | `[batch, features]` | Execute one time step with caller-owned state. |
| `TransformerEncoder` | `[batch, time, width]` | Stacked bidirectional attention, pre-normalization, GELU feed-forward blocks and final LayerNorm. |

`Sequential` composes these implementations with any existing DeepJ `Layer`, including
`Linear`, `FNN` and normalization layers. It deduplicates shared parameter tensors by
identity so existing optimizers update tied parameters once. Stack convolution,
activation and pooling layers to construct CNNs; compose recurrent layers to stack them.

## Tensor execution and gradients

All model arithmetic uses DeepJ tensor operations on the selected backend. Convolution
uses gathered image windows and matrix multiplication. Recurrent gates use existing
linear projections and tensor activations. Attention uses batched matrix products and
axis-specific softmax. Existing autograd supplies the gradients for these operations.

`TensorJoin` adds concatenation and stacking through existing backend copies and an
autograd slicing rule. It also supplies exact constant padding without multiplying
input values by zero. Shape indices, sequence-length masks, attention masks and random
dropout masks are constructed from host metadata; model activations are never downloaded
for forward or backward calculations. File I/O and explicit scalar loss reporting are
separate host transfers.

Selecting `MetalBackend` therefore uses the existing Metal operations for these layers.
Layers use the selected backend without switching it automatically. Convolution materializes
window tensors; its memory use grows with batch, output positions, channels and kernel
area. This implementation does not claim the speed of a dedicated fused convolution kernel.

Parameters accept the existing `ParameterFactory`, including optional compact storage.
BatchNorm statistics and attention score calculations use FLOAT32. Compact storage follows
DeepJ's existing promotion and FLOAT32-gradient policies.

## Composition and training mode

`Module` extends the existing `Layer` interface. Modules start in training mode;
`train()`, `eval()` and `setTraining(boolean)` propagate through their `Module` children.
Stateless layers compose through the same `Layer` interface.

`Activation` adapts existing activation functions; `Flatten` merges an inclusive range
of axes and preserves the batch axis by default. `Reshape` changes the complete shape
and accepts one inferred `-1` dimension.

`Dropout` uses inverted scaling during training and is the identity during evaluation.
A supplied `Random` or seed controls mask generation. Evaluation mode and
`GradMode.noGrad()` are independent: disabling autograd does not disable dropout.

`BatchNorm` normalizes channel axis 1, reducing the batch and spatial axes. Training
uses population variance for normalization and unbiased variance for the running estimate.
At least two values per channel are required during training. Evaluation uses the running
mean and variance; affine parameters and both running buffers are persisted by `NetworkWeights`.

`NdSupervisedTraining.trainer` provides first-axis minibatching for matrices, images and
sequences using the existing `Trainer`, losses and optimizers. Sampling uses backend
`indexSelect` operations. See [training](training.md) for the matrix, ND, and language-model training contracts.

## Recurrent state and masks

`Recurrent.run(input, initialState, lengths)` returns a `RecurrentResult` containing the
sequence and final `RecurrentState`. A null initial state creates zeros. A null lengths
array processes the full time dimension. Lengths range from zero to the padded time size.
Padded outputs are zero; state stops updating after each sample's valid length. A zero-length
sample retains its initial state. Nonempty batch and time dimensions are required.

State is passed explicitly, so independent calls do not carry hidden history. Passing the
returned state into the next chunk preserves gradient connections; `RecurrentState.detach()`
creates independent copies for truncated backpropagation, following DeepJ's existing
detach semantics. RNN uses tanh. GRU uses the reset-after formulation with reset, update,
and candidate gates. LSTM uses input, forget, candidate and output gates with cell memory.

## Transformer encoder masks

Inputs are embeddings. Apply the chosen positional representation before calling the
encoder. `TransformerEncoderConfig` defines model width, heads, feed-forward width, depth
and dropout; width must be divisible by the head count.

The optional `boolean[batch][time]` mask uses `true` for valid tokens. Padding is excluded
from attention keys and returned query positions are zero. Each sample must contain at
least one valid token. `meanPool(input, mask)` returns one vector per sample, dividing by
its valid-token count. Encoder blocks and attention can also be used independently.

## Weight archives

`NetworkWeights.save` and `NetworkWeights.load` use a separate versioned archive for
parameters and persistent buffers. Construct the intended architecture before loading.
The reader validates model type, entry names, shapes, dtypes, complete payload and end of
file before changing the network. Saves replace the destination through a temporary file.
Shared tensors are stored once; gradients are cleared after restoration.

This archive contains weights and persistent buffers. Architecture configuration, training
mode, optimizer state, recurrent state and dropout generator state are managed by the caller.
Existing language-model `.dj` and training-checkpoint formats are independent.

## Validation

Validation includes independent convolution and recurrent-equation references,
finite-difference input/parameter/state gradient checks, mask and shape validation,
training-convergence checks and exact weight/buffer round trips. Metal parity tests exercise
forward results, input and parameter gradients, and running buffers across all three dtypes
when an actual Metal device is available. Skipped Metal cases do not establish GPU correctness.

See [training](training.md), [saving and loading](persistence.md), and
[CPU and Metal execution](backends.md) for shared workflows.
