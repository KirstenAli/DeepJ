# Saving and loading

DeepJ provides persistence for network weights, language models, and resumable
language-model training. Choose the format that contains the state your application
needs. See [version compatibility](README.md#availability) when using an older dependency.

| Format/API | Contains | Restore contract |
|---|---|---|
| `NetworkWeights` archive | Model parameters and persistent module buffers, including BatchNorm statistics | Construct the intended architecture first; load validates type, entry names, shapes, and dtypes. |
| Language-model `.dj` checkpoint | Decoder weights and model settings | The model loader constructs the supported decoder architecture. |
| `TrainingCheckpoint` | Parameters, AdamW state, dataset cursor/random state, training progress, and schedule; model settings when supplied | Restore into matching model, optimizer, dataset, and schedule objects. |
| Model bundle | A DeepJ model checkpoint plus tokenizer files, configuration, and model card | Use DeepJ's language-model loaders and generation APIs. |

## Network weights and persistent buffers

`NetworkWeights.save(path, model)` writes a versioned archive through a temporary file.
`NetworkWeights.load(path, model)` restores into an already constructed network. The
reader validates the complete file, including its end, before applying tensor values.
Shared tensor identities are stored once. Parameter gradients are cleared after loading.

The archive preserves each tensor's stored dtype. Construct the target network with
matching dimensions, parameter names, sharing, and dtypes. Loading is not an implicit
precision conversion or a network-architecture reconstruction step.

Persistent buffers are separate from trainable parameters. `Module.buffers()` exposes
those buffers in traversal order; BatchNorm's running mean and variance are included.

Archives saved with a top-level `layers.FNN` or `nn.MLP` in earlier builds record
that class name. To migrate one, load it with the source revision that wrote it,
wrap the restored model in `Sequential`, and save again. Load that archive into a
matching `Sequential` containing `nn.FNN` with the same activations and parameter dtypes. Parameter names
and ordering are unchanged. Archives already rooted in `Sequential` remain compatible;
the package move does not alter language-model checkpoints.

## Caller-managed state

The network-weight archive contains parameters and persistent buffers. Keep these
items separately when they are needed:

- Architecture and layer configuration.
- Training/evaluation mode and optimizer/schedule state.
- Training counters, data cursor, and shuffle or sampling state.
- Recurrent hidden/cell state and dropout generator state.

Save and restore at a completed step boundary, outside a live backward graph. Use
[training checkpoint APIs](../src/main/java/io/github/kirstenali/deepj/persistence/TrainingCheckpoint.java)
for compatible resumable training state instead of treating a weights-only file as a
complete training checkpoint.

## Language-model files and bundles

`.dj` files and language-model bundles use their own model settings and tokenizer
contracts. They are independent of the `NetworkWeights` archive. A bundle published
on Hugging Face still requires DeepJ; it is not directly loadable by Python Transformers.
See [language-model loading](language-models.md#load-a-language-model).
