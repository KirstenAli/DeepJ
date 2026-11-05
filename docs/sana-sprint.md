# SANA-Sprint text-to-image

DeepJ implements SANA-Sprint 0.6B with its Gemma 2 text encoder, conditioned image transformer, stochastic consistency sampler, and DC-AE image decoder. Inference runs locally in Java through DeepJ tensors on CPU or Apple Metal. Python is used only to regenerate independent test fixtures.

The implementation lives in `io.github.kirstenali.deepj.nn.diffusion.sana`. It adds a standalone image launcher; the existing chat CLI and desktop application keep their current routing.

## Checkpoint

Use the complete [official SANA-Sprint 0.6B checkpoint](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers), revision `aa76e7f4f4928f378716b6716a2130fba3caf5b1`:

```sh
hf download Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \
  --revision aa76e7f4f4928f378716b6716a2130fba3caf5b1 \
  --local-dir checkpoints/sana-sprint
```

Keep the `tokenizer`, `text_encoder`, `transformer`, and `vae` folders with their JSON configuration and safetensors files. The image transformer has about 600 million parameters; the full pipeline also needs the larger Gemma 2 encoder and DC-AE decoder. The files occupy approximately 7.7 GB (7.2 GiB). Model weights are downloaded separately and are not bundled with DeepJ.

## Generate an image

Build the existing CLI distribution, then invoke the new image entry point:

```sh
mvn -Pcli -DskipTests package
java -Xmx8g --enable-native-access=ALL-UNNAMED \
  -cp target/deepj-cli.jar io.github.kirstenali.deepj.nn.diffusion.sana.SanaImageCli \
  --model checkpoints/sana-sprint --backend metal --output sana.png \
  'A red fox in a snowy forest, soft morning light, detailed wildlife photograph'
```

Use `--backend cpu` for CPU execution. The defaults are 1024 × 1024 pixels, two sampling steps, guidance 4.5, and seed 42. `--width`, `--height`, `--steps`, `--guidance`, and `--seed` are configurable. Dimensions must be multiples of 32 between 32 and 2048; sampling supports one through four steps. The checkpoint is designed for 1024-pixel images. Memory needs increase with resolution.

The pipeline loads and releases each model stage in sequence. Gemma weights use BF16 storage; the image transformer and decoder use FP32. During generation, DeepJ disables autograd graph recording. Image arithmetic stays on the selected tensor backend; small prompt features and final latents pass through host memory between model stages. Output is an RGB PNG.

A fixed seed repeats DeepJ's Gaussian noise sequence. Java and PyTorch use different random-number generators, so matching seed values do not produce identical noise across the two implementations.

## Java API

`SanaSprint.generate(prompt, options)` returns a `BufferedImage`. `SanaSprint.Options.defaults()` supplies the defaults above. Select DeepJ's tensor backend before calling it. An optional `Consumer<String>` constructor argument receives stage and sampling progress.

`Gemma2Encoder`, `SanaTransformer`, and `DcAeDecoder` expose tensor-level forward methods. Their operations use DeepJ autograd when graph recording is enabled. `SanaWeights` owns the parameters loaded from a stage directory and releases their device retention when closed. `parameters()` returns the parameters loaded so far; weights load on first use. Image generation is provided here; a pretrained-model fine-tuning workflow is not included.

`GroupedConv2D` adds grouped and depthwise, odd-kernel, same-padded NCHW cross-correlation. It reuses DeepJ's spatial patch gathering, tensor joins, matrix multiplication, and gradients. Spatial work is tiled to bound the patch allocation.

## Validation and references

Small deterministic checkpoints and expected outputs are generated independently with PyTorch 2.8.0, Transformers 4.51.3, and Diffusers 0.33.1 by `scripts/sana-reference.py`. Java tests compare Gemma sliding attention, masked image-transformer attention, DC-AE decoding, convolution outputs and gradients, and all one-to-four-step sampler schedules. The image transformer also has a finite-difference autograd check. Metal regression tests cover large finite GELU and tanh outputs and gradients in FLOAT32, FLOAT16, and BFLOAT16 storage.

```sh
mvn -Dtest='Sana*Test' test
mvn -Dtest='Sana*Test' -Dsana.test.metal=true test
mvn -Dtest=SanaCheckpointTest -Dsana.model=/absolute/path/to/sana-sprint test
```

The full official checkpoint was also exercised end to end on Apple M4 Metal at 1024 × 1024 with two sampling steps, producing an RGB image from a text prompt.

The last command checks the real checkpoint tokenizer against Hugging Face token IDs and prompt formatting. Metal checks require an Apple Silicon process with access to Metal.

Architecture and numerical behavior follow the official [SANA project](https://github.com/NVlabs/Sana), [Diffusers SANA-Sprint pipeline](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/pipelines/sana/pipeline_sana_sprint.py), and [Transformers Gemma 2 implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gemma2/modeling_gemma2.py). The instruction prefix follows the Diffusers pipeline (Hugging Face, Apache-2.0). The checkpoint and its Gemma component remain subject to their respective model licenses and terms; see the model card before redistribution.
