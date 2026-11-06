package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.ModelMemory;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Backbone;
import io.github.kirstenali.deepj.nn.transformer.gemma2.Gemma2Config;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.tokenizers.gemma.GemmaTokenizer;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Consumer;

/** Staged local SANA-Video inference: Gemma 2, video Transformer, then causal Wan video decoding. */
public final class SanaVideo {
    private final Path directory;
    private final Consumer<String> progress;

    public SanaVideo(Path directory) { this(directory, ignored -> {}); }

    public SanaVideo(Path directory, Consumer<String> progress) {
        this.directory = Objects.requireNonNull(directory);
        this.progress = Objects.requireNonNull(progress);
    }

    public record Options(int width, int height, int frames, int steps, float guidance, long seed) {
        public Options {
            dimension(width);
            dimension(height);
            if (frames < 1 || frames > 81 || (frames - 1) % 4 != 0) {
                throw new IllegalArgumentException("Video frames must be 4n+1, from 1 to 81");
            }
            if (steps < 1 || steps > 100 || !Float.isFinite(guidance) || guidance < 0) {
                throw new IllegalArgumentException("Expected 1-100 sampling steps and finite nonnegative guidance");
            }
        }
        public static Options defaults() { return new Options(832, 480, 9, 50, 6, 42); }

        private static void dimension(int size) {
            if (size < 16 || size > 2048 || size % 16 != 0) {
                throw new IllegalArgumentException("Video dimensions must be multiples of 16 from 16 to 2048");
            }
        }
    }

    public List<BufferedImage> generate(String prompt, Options options) throws IOException {
        if (prompt == null || prompt.isBlank()) throw new IllegalArgumentException("A video prompt is required");
        Objects.requireNonNull(options);
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open()) {
            ModelMemory.checkCancelled();
            SanaVideoSnapshot.validate(directory);
            Features[] captions = encode(prompt, options.guidance() > 1);
            return decode(sample(captions, options));
        } finally {
            Tensor.backend().releaseTemporaryResources();
        }
    }

    private Features[] encode(String prompt, boolean guidance) throws IOException {
        stage("Encoding video prompt with Gemma 2");
        var tokenizer = new GemmaTokenizer(directory.resolve("tokenizer"));
        Path stage = directory.resolve("text_encoder");
        try (var scope = TensorScope.open(); var weights = new VideoTextWeights(stage)) {
            var encoder = new Gemma2Backbone(Gemma2Config.read(stage.resolve("config.json")), weights::tensor);
            Features positive = encode(encoder, new SanaVideoPrompt(tokenizer, prompt, true));
            Features negative = guidance ? encode(encoder, new SanaVideoPrompt(tokenizer, "", false)) : positive;
            return new Features[]{positive, negative};
        }
    }

    private static Features encode(Gemma2Backbone encoder, SanaVideoPrompt prompt) {
        try (var scope = TensorScope.open()) {
            Tensor selected = encoder.forward(prompt.tokens(), prompt.valid()).indexSelect(1, prompt.selected());
            return new Features(ModelMemory.finite(selected, "Gemma video prompt"), selected.shape(), prompt.mask());
        }
    }

    private Tensor sample(Features[] features, Options options) throws IOException {
        stage("Loading SANA-Video transformer");
        Path stage = directory.resolve("transformer");
        try (var scope = TensorScope.open(); var transformer = SanaVideoTransformer.load(stage,
                VideoWeightCache.defaultBudget());
             var solver = FlowDpmSolver.read(directory.resolve("scheduler/scheduler_config.json"), options.steps())) {
            Tensor text = features[0].tensor(), negative = features[1].tensor();
            Tensor latent = noise(options);
            try (var pinned = ModelMemory.retain(text, negative)) {
                for (int step = 0; step < options.steps(); step++) {
                    stage("Video denoising step " + (step + 1) + "/" + options.steps());
                    latent = advance(transformer, solver, text, negative, features, latent, options.guidance());
                }
                return scope.escape(Tensor.from(ModelMemory.finite(latent, "SANA-Video"), latent.shape()));
            }
        }
    }

    private static Tensor noise(Options options) {
        int[] shape = {1, 16, 1 + (options.frames() - 1) / 4, options.height() / 8, options.width() / 8};
        int count = java.util.Arrays.stream(shape).reduce(1, Math::multiplyExact);
        float[] values = new float[count];
        Random random = new Random(options.seed());
        for (int i = 0; i < count; i++) { values[i] = (float) random.nextGaussian(); }
        return Tensor.from(values, shape);
    }

    private static Tensor advance(SanaVideoTransformer model, FlowDpmSolver solver, Tensor text,
                                  Tensor negative, Features[] features, Tensor latent, float guidance) {
        try (var scope = TensorScope.open(); var pinned = ModelMemory.retain(latent)) {
            Tensor flow = model.forward(latent, text, features[0].mask(), solver.timestep());
            if (guidance > 1) {
                try (var fixed = ModelMemory.retain(flow)) {
                    Tensor unconditional = model.forward(latent, negative, features[1].mask(), solver.timestep());
                    flow = unconditional.add(flow.subtract(unconditional).multiplyScalar(guidance));
                }
            }
            return scope.escape(solver.step(latent, flow));
        }
    }

    private List<BufferedImage> decode(Tensor latent) throws IOException {
        stage("Decoding frames with Wan video VAE");
        Path stage = directory.resolve("vae");
        try (var scope = TensorScope.open(); var weights = TensorWeights.open(stage, DType.FLOAT32)) {
            var decoder = new WanVideoDecoder(stage.resolve("config.json"), weights);
            return decoder.images(decoder.denormalize(latent));
        }
    }

    private void stage(String message) {
        ModelMemory.checkCancelled();
        progress.accept(message);
    }

    private record Features(float[] values, int[] shape, int[] mask) {
        Tensor tensor() { return Tensor.from(values, shape); }
    }
}
