package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Backbone;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Config;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.persistence.HuggingFaceConfig;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.tokenizers.gemma.GemmaTokenizer;
import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Random;
import java.util.function.Consumer;

/** Local SANA-Sprint text-to-image inference; model stages load and release their own weights in sequence. */
public final class SanaSprint {
    private final Path directory;
    private final Consumer<String> progress;

    public SanaSprint(Path directory) { this(directory, ignored -> {}); }
    public SanaSprint(Path directory, Consumer<String> progress) {
        this.directory = Objects.requireNonNull(directory);
        this.progress = Objects.requireNonNull(progress);
    }

    public record Options(int width, int height, int steps, float guidance, long seed, DType dtype) {
        public Options(int width, int height, int steps, float guidance, long seed) {
            this(width, height, steps, guidance, seed, DType.FLOAT32);
        }
        public Options {
            Objects.requireNonNull(dtype, "dtype");
            dimension(width);
            dimension(height);
            SanaScm.schedule(steps);
            if (!Float.isFinite(guidance) || guidance < 0) throw new IllegalArgumentException("Invalid guidance");
        }
        public static Options defaults() { return new Options(1024, 1024, 2, 4.5f, 42); }

        private static void dimension(int size) {
            if (size < 32 || size > 2048 || size % 32 != 0) {
                throw new IllegalArgumentException("Image dimensions must be multiples of 32 from 32 to 2048");
            }
        }
    }

    public BufferedImage generate(String prompt, Options options) throws IOException {
        if (prompt == null || prompt.isBlank()) throw new IllegalArgumentException("An image prompt is required");
        Objects.requireNonNull(options);
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open()) {
            ModelMemory.checkCancelled();
            PromptFeatures features = encode(prompt);
            Tensor latent = sample(features, options);
            return decode(latent, options.dtype());
        } finally {
            Tensor.backend().releaseTemporaryResources();
        }
    }

    private PromptFeatures encode(String text) throws IOException {
        stage("Encoding prompt with Gemma 2");
        var prompt = new SanaPrompt(new GemmaTokenizer(directory.resolve("tokenizer")), text);
        Path stage = directory.resolve("text_encoder");
        try (var scope = TensorScope.open(); var weights = TensorWeights.open(stage, DType.BFLOAT16)) {
            var encoder = new Gemma2Backbone(Gemma2Config.read(stage.resolve("config.json")), weights::tensor);
            Tensor selected = encoder.forward(prompt.tokens(), prompt.valid()).indexSelect(1, prompt.selected());
            return new PromptFeatures(ModelMemory.finite(selected, "Gemma"), selected.shape(), prompt.mask());
        }
    }

    private Tensor sample(PromptFeatures features, Options options) throws IOException {
        stage("Loading SANA-Sprint transformer");
        Path stage = directory.resolve("transformer");
        try (var scope = TensorScope.open(); var weights = TensorWeights.open(stage, options.dtype())) {
            var transformer = new SanaTransformer(stage.resolve("config.json"), weights);
            Tensor text = Tensor.from(features.values(), features.shape());
            Random random = new Random(options.seed());
            Tensor latent = SanaScm.noise(random, 1, 32, options.height() / 32, options.width() / 32)
                    .multiplyScalar(SanaScm.SIGMA);
            float[] times = SanaScm.schedule(options.steps());
            try (var pinned = ModelMemory.retain(text)) {
                for (int step = 0; step < options.steps(); step++) {
                    progress.accept("Denoising step " + (step + 1) + "/" + options.steps());
                    latent = step(transformer, text, features.mask(), latent, random, options, times, step);
                }
                float[] values = ModelMemory.finite(latent.divideScalar(SanaScm.SIGMA), "SANA");
                return scope.escape(Tensor.from(values, latent.shape()));
            }
        }
    }

    private Tensor step(SanaTransformer model, Tensor text, int[] mask, Tensor latent, Random random,
                        Options options, float[] times, int index) {
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(latent)) {
            ModelMemory.checkCancelled();
            Tensor input = SanaScm.modelInput(latent, SanaScm.time(times[index]));
            Tensor predicted = model.forward(input.to(options.dtype()), text.to(options.dtype()), mask,
                    SanaScm.time(times[index]), options.guidance()).to(DType.FLOAT32);
            Tensor denoised = SanaScm.denoise(latent, input, predicted, times[index]);
            Tensor next = index == options.steps() - 1 ? denoised
                    : SanaScm.advance(denoised, SanaScm.noise(random, latent.shape()), times[index + 1]);
            return scope.escape(ModelMemory.finish(next));
        }
    }

    private BufferedImage decode(Tensor latent, DType dtype) throws IOException {
        stage("Decoding image with DC-AE");
        Path stage = directory.resolve("vae");
        var config = new HuggingFaceConfig(stage.resolve("config.json"));
        try (var scope = TensorScope.open(); var weights = TensorWeights.open(stage, dtype)) {
            var decoder = new DcAeDecoder(stage.resolve("config.json"), weights);
            return SanaImages.rgb(decoder.forward(latent.to(dtype).divideScalar(config.decimal("scaling_factor"))));
        }
    }

    private void stage(String message) {
        progress.accept(message);
        ModelMemory.checkCancelled();
    }

    private record PromptFeatures(float[] values, int[] shape, int[] mask) {}
}
