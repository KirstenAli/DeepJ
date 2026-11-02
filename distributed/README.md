# DeepJ distributed training

An optional, separate artifact for synchronous data-parallel training with DeepJ
`0.13.0-alpha`. Each worker holds a full model and optimizer, computes its local
loss with DeepJ tensors and autograd, and exchanges FP32 gradients before the
optimizer update. Workers can select DeepJ's CPU or Metal backend.

This directory adds no dependency, hook, conditional, or extra call to the
existing library or ordinary training paths. It has its own Maven build; the root
project remains unchanged. Active distributed training adds network traffic and
CPU/GPU transfers. Speedup and useful model/batch sizes need measurements on the
target machines.

## API

All classes are in `io.github.kirstenali.deepj.distributed`.

| Class | Purpose |
| --- | --- |
| `TcpProcessGroup` | Rank-zero TCP coordinator, worker rendezvous, reductions, broadcasts and gathers. |
| `DistributedTrainer` | Initialization, local loss/backward, weighted gradient averaging, global clipping and optimizer updates. |
| `TensorObjective` | Adapter for existing `Layer` networks and a scalar mean `LossFunction`. |
| `CausalObjective` | Adapter for existing `CausalLM` models, including masked sequence losses. |
| `TrainingObjective` | Custom objective contract for any `Trainable` model. |
| `ShardSampler` | Deterministic, shuffled, disjoint sample indices and resumable epoch/cursor state. |
| `TensorShard`, `CausalShard` | Sample sources that retain complete tensors or token sequences and their masks. |
| `DistributedCheckpoint` | Per-rank weights, persistent buffers, AdamW state, progress and worker-state files. |
| `WorkerState` | Snapshot, validation and restoration contract for custom data, RNG or recurrent state. |

`DistributedTrainer.trainer()` also exposes DeepJ's existing `Trainer` interface.
The model, parameter objects, autograd engine, tensor backend and optimizer are
the existing DeepJ implementations. No network implementation is replaced.

## Workers and synchronization

Run one JVM per worker and select its tensor backend before constructing tensors
or models. DeepJ's backend selection is process-wide. Assign every worker the same
world size and a unique rank from zero through `worldSize - 1`.

Rank zero calls `TcpProcessGroup.coordinator(address, workers, session, timeout)`.
Other ranks call `TcpProcessGroup.connect(address, rank, workers, session, timeout)`
using rank zero's reachable address and the same session label. Rank zero also
trains. The coordinator accepts peers on the first collective. The session label
checks job identity; TCP traffic is unencrypted and the label is not authentication.
Use a trusted network or an authenticated tunnel.

All workers must execute the same collective calls in the same order. Each call
has an absolute deadline, including rendezvous, reads and writes. A disconnect,
timeout, incompatible layout or reported worker failure makes the group terminal.
Restart the job from a completed checkpoint after resolving the failure.

Initialization checks parameter names, shapes, dtypes, frozen flags, model and
optimizer classes, clipping and communication settings. DeepJ AdamW settings must
match, and its counter must start at zero. Rank zero broadcasts initial parameters
and persistent buffers. Parameter and buffer identities, order and frozen flags
must remain fixed for the job. Use the same architecture and loss configuration
on every rank; parameter metadata cannot describe every forward-pass setting.
Custom optimizers must use identical configuration
and initial state on every rank; their private state is not inspected.

## Losses, gradients and data

`TrainingObjective.evaluate(...)` returns a `WeightedLoss`: a differentiable scalar
**sum** and a nonnegative contribution count. `WeightedLoss.mean(mean, count)`
converts a local mean. Unequal and short batches use their actual counts, rather
than treating every worker as equally weighted. An exhausted worker returns
`WeightedLoss.empty()` and still participates in synchronization.

`TensorObjective` expects a loss that averages samples and uniformly sized feature
axes. `CausalObjective` preserves DeepJ's existing mean-per-sequence objective:
each sequence's cross-entropy uses its own response-mask denominator, and sequence
means are then averaged across the actual global batch. A custom token-weighted
objective can instead supply its own sum and token count.

The trainer clears gradients, evaluates and backpropagates locally, sums gradients,
divides by the global contribution count, clips the synchronized gradient and calls
the existing optimizer. Missing local gradients contribute zeros. FP32 gradient
communication supports FLOAT32, FLOAT16 and BFLOAT16 parameter storage. Non-finite
losses or synchronized gradients stop training before the optimizer step.

`ShardSampler` assigns each shuffled sample to exactly one rank without padding or
duplication. Every rank needs the same dataset ordering, sample count, shuffle seed
and epoch. `TensorShard` selects complete samples; `CausalShard` preserves complete
sequence inputs, targets and masks. Call `setEpoch(...)` explicitly for each epoch.
For equal local batch sizes, run `stepsPerEpoch(batchSize)` steps on every rank,
including ranks that exhaust earlier. With different local batch sizes, agree on
the maximum local step count across ranks. A step where all shards are exhausted
ends the group with an error.

The family-independent parameter protocol applies to feed-forward networks,
convolutions, recurrent networks, autoencoders, encoders and complete decoder
models. It does not make every operation equivalent to processing one combined
batch: batch-dependent normalization and worker-local randomness retain their
local semantics. Floating-point results can also differ across backends and batch
groupings.

## Persistent buffers and worker state

`BufferPolicy.BROADCAST_FROM_ROOT` is the default: persistent buffers start from
rank zero and are broadcast again after each update. `BufferPolicy.LOCAL` shares
initial buffers and then lets each worker maintain its own values.

BatchNorm computes training statistics from each worker's local batch with either
policy. Broadcasting running buffers does not provide synchronized BatchNorm over
the global batch.

The built-in shard sources checkpoint their epoch and cursor automatically.
`CausalObjective` also supports DeepJ's `StatefulBatchSource` random-state API.
Custom sources or objectives should expose `WorkerState` when exact continuation
needs additional RNG, hidden-state or data-loader state. `validate(...)` must check
the snapshot without changing live state. The default worker state is empty; it
cannot capture undisclosed model or source state.

## Checkpoints

Every worker calls `DistributedCheckpoint.save(trainer, directory)` at the same
step. Each rank writes `rank-N.djdist` on its own filesystem; a shared directory is
also supported because filenames include the rank. A checkpoint contains tensor
parameters and persistent buffers, all AdamW moments and master weights, the
current learning rate and counter, training progress and that rank's worker state.
Files have a versioned header, shared save identifier and SHA-256 checksum.

Every worker must finish the save successfully. Keep only complete checkpoint
cohorts; a partial save can leave files from different saves in the directory,
which are rejected at load even if their training steps match.
Use a fresh directory per save when retaining earlier checkpoints. Individual
files are atomically replaced where the filesystem supports it, with a replacement
fallback on other filesystems. Completion is coordinated across ranks; it is not a
cross-machine filesystem transaction.

To resume, construct the same architecture and initial optimizer configuration
with a fresh AdamW, create a group with the same world size and rank assignments,
and have every rank call `DistributedCheckpoint.load(...)`. Full file checks,
layout checks, worker-state validation and a collective validation vote precede
restoring live tensors. AdamW is currently the supported checkpointed optimizer.
The format is separate from DeepJ's existing language-model checkpoints.

A worker failure during an optimizer update or live restoration can leave some
workers changed; that terminal job must restart. There is no optimizer rollback,
elastic membership or automatic worker rejoin.

## Build and verification

Install the unchanged core library into the local Maven repository, then build
this artifact independently:

```sh
mvn --batch-mode install -Pfull-coverage
mvn --batch-mode -f distributed/pom.xml verify
```

The portable suite runs real worker JVMs over process pipes using the production
coordinator, framing, reductions and deadlines. It compares 15 network families
and three storage dtypes against the existing single-worker training APIs,
including unequal batches, an exhausted rank, clipping and AdamW state. It also
checks checkpoint continuation, disjoint shards, persistent-buffer policies,
non-finite gradients, backward failures, protocol mismatch and worker termination.

On a runner that permits listening sockets, switch those subprocess checks to
production TCP and include rendezvous mismatch checks:

```sh
mvn --batch-mode -f distributed/pom.xml verify -Ddeepj.distributed.tcp=true
```

On an Apple Silicon runner with a physical Metal device, require GPU execution as
well as TCP:

```sh
mvn --batch-mode -f distributed/pom.xml verify \
  -Ddeepj.distributed.tcp=true -Ddeepj.distributed.requireMetal=true
```

Requesting Metal fails if a physical device is unavailable; it does not substitute
CPU execution. The additional checks require GPU-backed parameter gradients and
exercise training and checkpoint continuation in separate Metal worker JVMs.
Without these properties, the TCP rendezvous and Metal-specific tests are reported
as skipped. A portable pass is not a TCP, Metal or multi-machine qualification.

The separate `Distributed training` workflow runs TCP checks on pull requests and
can run the Metal checks through `workflow_dispatch` on a self-hosted runner
labelled `deepj-metal`. These tests use one host; actual multi-Mac throughput,
reliability and mixed-backend behavior still require measurements on the target
machines. FP32 gradients are staged through host memory in configurable buckets
(default 262,144 elements, or 1 MiB). Rank zero coordinates all traffic, so network
bandwidth and coordinator memory/throughput constrain scaling.
