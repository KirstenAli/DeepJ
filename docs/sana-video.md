# SANA-Video text-to-video

DeepJ implements SANA-Video 2B with the existing Gemma 2 text encoder, a video Transformer with 3D rotary attention and temporal convolutions, a flow-prediction DPM-Solver++ sampler, and the Wan 2.1 causal video decoder. Network arithmetic uses DeepJ tensors on CPU or Apple Metal. Java inference does not call Python.

The implementation is in `io.github.kirstenali.deepj.nn.diffusion.sanavideo`. The `deepj video` command and desktop app generate videos and save H.264 MP4 files using JCodec in Java. The standalone launcher also remains available.

## Checkpoint

Download the complete [official 480p checkpoint](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers), revision `db5f398b13ca086d09a50ce156c20527773841b1`:

```sh
hf download Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --revision db5f398b13ca086d09a50ce156c20527773841b1 \
  --local-dir checkpoints/sana-video
```

Keep `model_index.json` and the `tokenizer`, `text_encoder`, `transformer`, `vae`, and `scheduler` directories. The complete checkpoint occupies about 14 GB, including the larger Gemma encoder. The 2B label describes the video Transformer. Weights are downloaded separately and retain their original model terms, including Gemma's terms.

For sharded Diffusers Transformer weights, DeepJ creates `transformer/model.safetensors.index.json` as a copy of the official index. Shard names and parameter mappings remain unchanged. This lets the video pipeline reuse the existing shared checkpoint loader.

## Generate a short clip

Build the CLI distribution and generate a short MP4:

```sh
mvn -Pcli -DskipTests package
java -Xmx2560m --enable-native-access=ALL-UNNAMED \
  -jar target/deepj-cli.jar video \
  --model checkpoints/sana-video --backend metal \
  --width 832 --height 480 --frames 9 --steps 50 --fps 16 --output fox.mp4 \
  'A red fox slowly walking through a snowy pine forest, gentle falling snow, steady camera'
```

The library, CLI, and desktop default to a short nine-frame clip at 832 × 480, 50 sampling steps, guidance 6, seed 42, and FP32 execution. The CLI and desktop use 16 FPS, and the CLI saves to `sana-video.mp4`. The upstream 480p configuration uses 81 frames; request `--frames 81` explicitly for that length. Higher resolution, more frames, and more steps increase memory use and generation time. Full 81-frame generation remains unvalidated on the 16 GB Mac. Reduced spatial sizes can produce numerically unstable attention in both the reference model and DeepJ; they are execution probes, not quality demonstrations.

Choose `--backend cpu` for CPU execution. CPU tensors live in the JVM heap, so increase `-Xmx` from the Metal example to accommodate the checkpoint and activations. Width and height must be multiples of 16 from 16 to 2048; frame counts must be `4n+1` from 1 to 81. The CLI accepts `--guidance`, `--seed`, and `--steps` (1–100). The output path ends in `.mp4`. MP4 uses the requested frame rate and contains no audio. The CLI also accepts `.gif` for an optional animation export; GIF uses a reduced color palette and centisecond timing.

A fixed seed repeats DeepJ's Gaussian noise sequence. Java and PyTorch use different random-number generators, so equal seeds do not imply equal initial latents between engines. The exact requested dimensions are used; the launcher does not apply Diffusers' automatic resolution binning.

## Desktop playback and saving

```sh
JAVA_TOOL_OPTIONS="-Xmx2560m" mvn compile javafx:run
```

Start the desktop app and choose **Open video model folder…** from the model menu. Select the complete
SANA-Video snapshot, then describe a scene and its movement in the composer. The settings chip opens
resolution, frame count, sampling steps, FPS, seed, guidance, and precision controls.
FP32 is the default. FP16/BF16 select compact Gemma and video-denoiser execution;
the sampler and Wan frame decoder continue to use FP32. The published
[Diffusers example](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/sana_video/pipeline_sana_video.py)
also combines a BF16 text encoder/denoiser with an FP32 Wan decoder; its sampler keeps FP32 latent state.

The resolution menu also offers **192 × 128 · Test only** for execution checks.
Its output does not establish full-resolution video quality.

The result has play/pause and a seek bar. **Save video** opens the native file picker and saves an MP4;
its compact style matches **Save image**. Preview playback uses the generated frames so it does not depend
on the embedded browser's native video codecs. MP4 encoding runs in the background. Stop generation waits
for worker cleanup before enabling another request, and starting a new conversation removes temporary previews.

## Execution and tensor APIs

The pipeline loads Gemma, samples video latents, and decodes video in separate stages. By default the text encoder, video Transformer, sampler, and video decoder use FP32
working weights and activations. `--dtype float16` or `--dtype bfloat16` selects
compact text-encoder and Transformer weights and activations. The sampler and video
decoder retain FP32. FP16 query/key projections, RMS statistics,
linear-attention denominators, and gated residual accumulation use FP32 to avoid
overflow and preserve the stabilizing epsilon. Projections return to FP16 after
normalization; convolution and other linear layers retain FP16 execution.
Gemma encodes the positive and unconditional captions together. Small previews
also batch the guided Transformer predictions; larger grids retain bounded
sequential guidance execution.

Gemma and the video Transformer use the same budgeted weight loader. Each caches its weights when their original checkpoint storage fits the cache budget; otherwise it streams groups of weights. Cached values retain their original storage dtype, and active groups are converted to the selected working dtype for calculation. Compact checkpoint files remain supported and are never overwritten. Expanding a BF16 checkpoint to FP32 preserves its stored values; it cannot recover information lost before that checkpoint was saved. The decoder uses its stage loader.

The automatic Metal weight-cache budget is half the managed Metal buffer budget, capped at 4 GiB. The default 6 GiB managed budget therefore permits a 3 GiB weight cache and reserves the remaining space for activations and expanded working weights. On CPU, the automatic cache budget is one quarter of the maximum JVM heap. Override the cache using `-Ddeepj.video.weightCacheBytes=BYTES`; zero forces streaming. The managed Metal guard continues to enforce the total native buffer budget.

`SanaVideo.generate(prompt, options)` returns `List<BufferedImage>`. `VideoMp4.write(frames, path, fps)` exports H.264 MP4; `VideoGif.write(...)` remains available for animated GIFs. `SanaVideoTransformer` accepts matching FP32, FP16 or BF16 inputs and weights;
`WanVideoDecoder` exposes an FP32 tensor-level forward method; constructing the Transformer with `TensorWeights` supports DeepJ autograd. `SanaVideoTransformer.load(directory, cacheBytes)` selects the weight-cache budget, and `streamed(directory)` disables caching.
`load(directory, cacheBytes, dtype)` selects compact execution. The seven-argument
`SanaVideo.Options` constructor accepts a final `DType`; existing constructors retain FP32. Loaded Transformers own their weights, implement `AutoCloseable`, and require `GradMode.noGrad()`.

End-to-end generation disables gradient recording and releases per-video causal caches. The Wan decoder processes the first latent frame into one RGB frame, then each later latent frame into four RGB frames. A VAE encoder and pretrained fine-tuning workflow are not included.

Video stage boundaries release temporary buffers while retaining the tensors needed by later steps. During inference, causal VAE state stays on the selected backend. After temporal upsampling, the decoder processes frames individually through the remaining stages and converts finished frames directly to RGB. Temporal convolution reads only the required history slices, and the cache retains only the required tail. This avoids large joined-history tensors and a second full floating-point video. The differentiable tensor path retains its graph.

Video Transformer inference also retires attention intermediates before feed-forward work. Longer sequences
stage self-attention projections and split cross-attention into independent groups of query rows while retaining
the complete caption context for each row. Feed-forward spatial work processes one latent frame at a time
for longer clips, then applies the same temporal convolution to their combined results. These inference paths
bound temporary storage without changing the temporal kernel or the differentiable calculation path.

At 832 × 480, the decoder's required FP32 causal histories alone occupy approximately 3.52 GiB once filled. Each temporal convolution keeps at most two past input frames; this storage does not increase with clip length. The recorded full-resolution decode held its managed allocation constant from RGB frame three through nine. A separate tiny native stress check held allocation constant through 200 cache updates and returned to its initial allocation after closing. This establishes bounded causal-cache retention; total process memory also includes other activations, checkpoint transfers, output images, and JVM storage. macOS may use swap under system-wide memory pressure, independently of whether inference runs in Java or Python.

The video launchers and desktop video jobs default to a 6 GiB budget for managed Metal buffers. Override it explicitly with `-Ddeepj.metal.maxBufferBytes=BYTES` before `-jar` or the class name. The budget is not a total process-memory limit; JVM storage and native library workspace also use memory. Desktop video generation runs in a separate worker JVM so its budget is configured before Metal initializes. The chat backend stays in the desktop process. Cancelling generation terminates the video worker before another job is enabled. Existing chat and training commands retain their own settings. Library callers can set the same property before initializing Metal.

## Validation

The [precision audit](benchmarks/precision-audit-2026-11-08/REPORT.md) records
short FP32/FP16/BF16 pretrained preview comparisons, output differences and timings.
The [multi-step follow-up](benchmarks/video-lifetime-2026-11-08/REPORT.md) checks
sampler-state retention, corrects an MPS reference-scheduler issue and records
complete FP32/FP16/BF16 frame comparisons and Python-only rounding controls.

Independent small checkpoints and expected outputs are generated by `scripts/sana-video-reference.py` with PyTorch 2.8.0, Diffusers 0.42.1, and Transformers 5.22.1. Tests compare FP32 video Transformer outputs, masked caption attention, 3D rotary mixing, causal convolution outputs and gradients, Wan decoder outputs and temporal gradients, and every sampler step. They also check cached and streamed execution, compact checkpoint loading, prompt token IDs, cancellation, repeat generation, animated GIF frame counts and timing, and MP4 decoding, frame counts, timing, cancellation, and saving. Desktop tests exercise playback, seeking, Thinking mode, and the native save-picker bridge.

```sh
mvn -Dtest='SanaVideo*Test,TensorFileVideoTest,DepthwiseConv2DTest' test
mvn -Dtest='SanaVideo*Test,TensorFileVideoTest,DepthwiseConv2DTest' -Dsanavideo.test.metal=true test
mvn -Dtest=SanaVideoCheckpointTest -Dsanavideo.model=/absolute/path/to/sana-video test
```

The optional checkpoint test checks real tokenizer IDs and checkpoint metadata. Metal tests require an Apple Silicon process with GPU access. Python is used only to regenerate test fixtures; `--tokenizer /absolute/path/to/checkpoint/tokenizer` regenerates the optional tokenizer oracle.

## Pretrained comparison

A matched pretrained comparison on a 16 GB Apple M4 uses **832 × 480, nine frames, 50 sampling steps, guidance 6, and seed 42**. Both engines receive identical saved token IDs, attention masks, and starting noise. Each independently computes Gemma features and feeds its own predictions back into its sampler. Java and Python seeds alone are not used to establish matching inputs.

The local Transformer checkpoint was rounded to BF16 for storage. Both engines expand those same stored values to FP32; this comparison does not cover the original, unrounded FP32 Transformer snapshot. Gemma retains the original BF16 checkpoint values, and the VAE retains FP32 values. All calculations in this comparison use FP32.

RGB differences below use the 0–255 channel scale:

| Comparison                            | Mean difference | Maximum difference |
| ------------------------------------- | --------------- | ------------------ |
| Complete independent pipeline         | 0.00224         | 1                  |
| Decoder given identical final latents | 0.0000493       | 1                  |

The final latent RMSE is `0.0000420`. These results establish close agreement for the matched checkpoint and inputs; they do not imply pixel-identical output across engines.

Observed DeepJ stage times and managed Metal peak from one validation run:

| Prompt encoding | 50 sampling steps | Decode nine frames | Peak managed Metal buffers |
| --------------- | ----------------- | ------------------ | -------------------------- |
| 32.1 s          | 83.8 min          | 252.7 s            | 5.84 GiB                   |

These are single observations, not a repeated performance benchmark. The pretrained text encoder and Transformer stream under a 3 GiB weight-cache budget. The decoder determines the overall managed peak. The 6 GiB managed buffer guard is not a total process-memory measurement; JVM RSS and unified-memory buffers can overlap and must not be added together.

The independent reference uses PyTorch 2.8.0, Diffusers 0.42.1, and Transformers 5.22.1. It offloads weights between CPU and MPS and runs the scheduler on CPU. To fit the real VAE, it schedules decoder blocks framewise after temporal upsampling and tiles local Conv3D output rows with full spatial halos. A control against the unmodified pretrained decoder on nine and 21 frames differs by at most `0.00000103` in normalized pixels. The reference uses a 6 GiB MPS driver allocation cap for encoding/sampling and 8 GiB for decoding; driver allocation and DeepJ's managed buffer counter measure different things. These runs are correctness checks, not a Java-versus-Python speed comparison.

Nine-frame full-resolution FP32 generation completed and closely matches the independent Python pipeline for the matched local checkpoint. At 81 frames, a one-step FP32 sampling check completed in 536 seconds with a peak of 5.92 GiB of managed Metal buffers. This is a capacity check; complete 81-frame decoding and 50-step generation quality remain unvalidated. Reduced 192 × 128 execution probes can amplify tiny numerical differences because some attention denominators approach the upstream safeguard; they are not the basis for the full-resolution results above.

Regenerate the small fixtures using the pinned Python versions above:

```sh
python scripts/sana-video-reference.py
javac -cp target/classes -d target/reference-inputs scripts/SanaVideoFixtureInputs.java
java -cp target/classes:target/reference-inputs \
  io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideoFixtureInputs
python scripts/sana-video-trajectory-reference.py
python scripts/sana-video-pipeline-reference.py
npx prettier --write "src/test/resources/sana-video/**/*.json"
```

Compile DeepJ before the Java input exporter. The exporter supplies inputs only; all expected predictions and pixels are calculated independently by Python. The Python sampling reference runs its scheduler on CPU to avoid the PyTorch 2.8.0 MPS CPU-scalar-view issue observed during the investigation. Rotary frequency buffers retain their constructor precision when checkpoint weights are loaded.

## References and attribution

Architecture and sampling behavior follow the [SANA-Video project](https://nvlabs.github.io/Sana/docs/sana_video/), [Diffusers video Transformer](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/transformers/transformer_sana_video.py), [Wan video autoencoder](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/autoencoders/autoencoder_kl_wan.py), and [DPM-Solver++ scheduler](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py). Adapted portions, including the prompt instruction, retain Apache-2.0 attribution in `META-INF/licenses/sana-video-NOTICE.txt` and the accompanying license.

## Compact execution checks

`scripts/compact-model-reference.py` generates independent FP16/BF16 Gemma and
SANA denoiser outputs. CPU and Metal tests cover those predictions, matching input
and weight dtypes, streamed compact weights, and tiny two-step pipelines through
FP32 sampling and frame decoding. Full pretrained 16-bit video quality has not
been established by these small checks; the full-resolution comparisons above
measure FP32. Compact modes are explicit selections rather than new defaults.

[Apple M4 demo checks](benchmarks/media-demo-2026-11-07/REPORT.md#sana-video-small-execution-check)
record a small pretrained 192 × 128, five-frame, one-step execution check and
its stage timings. No long video generation was run for that check.
