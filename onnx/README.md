# DeepJ ONNX

Optional ONNX graph import and execution using DeepJ tensors, autograd and the selected CPU or Metal backend. The importer executes the graph's operations, so support depends on its operators and data types rather than a list of model architectures.

This is a separate `deepj-onnx` artifact. Existing DeepJ model, checkpoint, CLI, desktop and training code is unchanged.

## Build

Use Java 20 and install the DeepJ core from the same checkout before building the optional artifact:

```sh
mvn install
mvn -f onnx/pom.xml verify
```

The artifact coordinates are `io.github.kirstenali:deepj-onnx:0.1.0-alpha-SNAPSHOT`. Add it alongside DeepJ after installing it with `mvn -f onnx/pom.xml install`. Maven generates the Java protobuf bindings from the pinned official ONNX schema.

## Code quality

`mvn -f onnx/pom.xml verify` enforces the shared Java standards in production and test code: 20 non-empty lines and 15 executable statements per function, complexity and nesting limits, 120-character lines, and copy/paste detection from 100 tokens. Spotless also checks a pinned four-space Java format. Use `mvn -f onnx/pom.xml spotless:apply` to apply that format before verification.

JaCoCo enforces at least 85% line coverage and 70% branch coverage for the handwritten ONNX implementation. The generated protobuf bindings are the only Java code excluded from formatting, style and coverage analysis; they are generated from the pinned official schema. PMD and duplication checks fail the build on violations or analysis errors.

## Import and execution

`OnnxModel.load(Path)` reads a `.onnx` graph and its embedded or external weights. Choose the DeepJ tensor backend before loading, and keep that backend selected while using the model. Close the model when finished to release its retained weight buffers.

`OnnxModel.inspect(Path)` reports the graph's named inputs, outputs, imported operator-set versions and static compatibility problems. Loading performs these checks before allocating weights. Runtime checks validate actual input shapes, symbolic dimensions, element types, operator arguments and declared output shapes.

`forward(Tensor)` handles graphs with one floating input and one floating output. `run(Map<String, OnnxValue>)` handles named inputs, multiple outputs, integer IDs, boolean conditions and explicit recurrent state. Both execute the same graph.

`OnnxValue.of(Tensor)` wraps an existing DeepJ tensor without copying it. `integers(long[], int...)` creates exact INT64 inputs; `booleans(boolean[], int...)` creates BOOL inputs. `integerTensor(int, long[], int...)` accepts another supported ONNX integer element-type number. Floating outputs expose their DeepJ tensor through `tensor()`; integer and boolean outputs expose exact values through `longs()`.

Tokenization, image preprocessing, generation loops and diffusion samplers belong to the application using the imported graph. Importing a graph does not register a model in the existing chat or media interfaces.

## Training and persistence

`OnnxModel` implements DeepJ's existing `Layer` interface. Loading creates inference weights without trainable parameter wrappers or gradient buffers. Call `trainable(String...)` with floating initializer names to select the weights to optimize. Repeated selection of a name reuses the same `Parameter`.

Compute a differentiable loss from the output tensor, call `backward()`, and update `parameters()` with an existing DeepJ optimizer. Input gradients work when the input tensor requires gradients. Integer/boolean calculations and branch selection do not have gradients; floating arithmetic inside the selected branch uses DeepJ autograd.

ONNX inputs and attributes control Dropout and BatchNormalization training behavior. Selecting trainable weights does not rewrite an exported inference graph. Running statistics returned by a training graph are explicit outputs.

DeepJ's existing `NetworkWeights.save/load` API saves and restores the selected parameters. Reload the same ONNX graph, select the same names in the same order, then restore the archive. Keep the original ONNX graph and its external data; the archive does not replace the graph or store optimizer state.

## Compatibility

The parser and versioned operator contracts are pinned to [ONNX v1.23.2](https://github.com/onnx/onnx/tree/v1.23.2). The loader accepts IR versions 3–14 and registered standard operators through opset 28; each operator also has its own minimum version. Attribute availability follows the schema for the model's imported opset.

| Operations | Implementations |
| --- | --- |
| Arithmetic | Add, Sub, Mul, Div, Neg, Abs, Exp, Log, Sqrt, Reciprocal, MatMul, Gemm, Pow, Sum, Mean, Min, Max |
| Activations | Relu, Sigmoid, Tanh, Softplus, Softsign, Erf, Gelu, LeakyRelu, PRelu, Elu, Selu, HardSigmoid, HardSwish, Mish, Clip |
| Normalization | LayerNormalization, InstanceNormalization, BatchNormalization, GroupNormalization, Dropout |
| Convolution | Conv and ConvTranspose with spatial dimensions, groups, dilation, strides and padding |
| Pooling | MaxPool, AveragePool, GlobalMaxPool, GlobalAveragePool |
| Shapes and indexing | Identity, Constant, ConstantOfShape, Shape, Size, Reshape, Flatten, Transpose, Expand, Concat, Split, Squeeze, Unsqueeze, Tile, Slice, Gather, GatherElements, GatherND, Pad, Resize, Upsample |
| Reductions | ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceSumSquare, Softmax, LogSoftmax, ArgMax, ArgMin, TopK |
| Types and conditions | Cast, CastLike, Range, Equal, Greater, Less, GreaterOrEqual, LessOrEqual, And, Or, Xor, Not, Where |
| Recurrent networks | RNN, GRU and LSTM, including reverse/bidirectional execution, sequence lengths and explicit hidden/cell state |
| Graph composition | If, bounded Loop, model-local functions and explicitly registered custom-domain operators |

Floating neural calculations use FLOAT32, FLOAT16 or BFLOAT16 tensors. Gradients use DeepJ's FLOAT32 accumulation. Integer and boolean values use exact host storage, including INT64 values above the FLOAT32 exact range. UINT64 values must fit a signed Java `long`; integer arithmetic overflow is rejected. Arithmetic support for integers covers Add/Sub/Mul/Div and the shape, indexing, comparison, cast and selection paths. Integer reductions include sum, product, extrema, L1 and sum-of-squares. Other arithmetic and neural calculations require floating tensors.

`Pow` currently requires a constant scalar floating exponent. Resize/Upsample implement nearest and linear interpolation without antialiasing, with stretch aspect ratio and the supported coordinate modes. ConvTranspose uses padding and output-padding attributes; `output_shape` is unsupported. MaxPool uses row-major indices. LayerNormalization uses FLOAT32 statistics. Local functions must import operator-set versions compatible with the containing model; conflicting versions are rejected.

FLOAT64, strings, sparse tensors, tensor sequences, quantized graph operations, Scan and unregistered custom operators are outside the current coverage. Unsupported operators, versions and attributes produce diagnostics. A model's architecture name alone does not establish compatibility.

Floating calculations dispatch through DeepJ's selected backend. Shape logic, exact integer operations, comparisons and selection indices use host arrays; value-dependent comparisons and selection can require device readback. Convolution and recurrent operations compose existing tensor operations and gradients.

## Resource bounds

`OnnxOptions` configures the maximum protobuf bytes, retained weight payload and loop iterations. Defaults are 256 MiB, 2 GiB and 10,000 iterations. Raise these explicitly for larger graphs that fit the machine. These bounds do not estimate intermediate activation memory.

External tensor files must resolve inside the model's directory. Absolute paths, parent-directory segments, escaping symlinks, invalid ranges and truncated data are rejected. Keep external data next to its `.onnx` file when moving a model.

## Validation

The checked-in fixtures are small deterministic graphs evaluated independently by ONNX Runtime **1.31.1 CPU** with ONNX **1.23.2**. They cover MLPs, 1D/2D/3D grouped convolution, transposed convolution, pooling, padding, interpolation, normalization, RNN/GRU/LSTM, an attention block, local functions, control flow, exact integers, external data and FP16 graphs.

The Java suite compares these outputs on CPU or Metal. MLP, convolution, GRU and attention gradients are compared against finite differences calculated by ONNX Runtime. Additional tests check optimizer updates, parameter archive restoration, malformed graphs, typed/raw weights, external-file boundaries and configured resource bounds. These checks establish behavior for the included graphs; they are not a benchmark or validation of arbitrary pretrained models.

Run the same suite on a physical Metal GPU:

```sh
mvn -f onnx/pom.xml -Donnx.test.metal=true verify
```

This option fails if Metal is unavailable. The ONNX CI workflow also provides an optional self-hosted Metal job. The ordinary CPU suite reads the checked-in oracles and needs no Python installation.

The Python files under `tools/` are development-only reference generators. Maven tests read their checked-in outputs; neither building the Java artifact nor running an imported model starts Python or ONNX Runtime.

To regenerate reference data, use an isolated Python environment with the versions in `tools/requirements.txt`, then run `generate_fixtures.py`, `generate_gradients.py` and `generate_schemas.py` from `onnx/tools`. Python and ONNX Runtime are used for reference generation; Java execution uses DeepJ.

See `NOTICE` and `LICENSE-ONNX.txt` for the official schema's provenance and license.
