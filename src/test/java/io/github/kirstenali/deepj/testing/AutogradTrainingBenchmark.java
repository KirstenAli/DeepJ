package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.nova.DeepJNova;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.Trainer;

import java.util.Arrays;

/** Uses only APIs shared with the manual-backprop baseline; no dataset or checkpoint access. */
public final class AutogradTrainingBenchmark {
    private AutogradTrainingBenchmark() {}

    public static void main(String[] args) {
        if (!MetalBackend.isAvailable()) throw new IllegalStateException("This benchmark requires a Metal device");
        Tensor.setBackend(new MetalBackend());
        try {
            benchmark(args[0], Integer.parseInt(args[1]));
        } finally {
            Tensor.backend().releaseResources();
        }
    }

    private static DecoderOnlyModel model(String profile) {
        return switch (profile) {
            case "prism" -> new DeepJPrism(new DeepJPrismConfig(
                    16_384, 1_024, 768, 12, 10, 2_112, 384, 192, 0.2f, 1), 100);
            case "nova" -> new DeepJNova(new DeepJNovaConfig(
                    16_384, 1_024, 576, 9, 3, 24, 1_664, true, 0.2f, 1), 100);
            default -> throw new IllegalArgumentException("Unknown benchmark profile: " + profile);
        };
    }

    private static void benchmark(String profile, int batchSize) {
        if (batchSize < 1) throw new IllegalArgumentException("Batch size must be positive");
        DecoderOnlyModel model = model(profile);
        Batch batch = batch(batchSize);
        Trainer trainer = CausalLMTraining.trainer(model, ignored -> batch, 1e-4f);
        rounds(trainer, batchSize, "warmup", 4);
        double[] seconds = rounds(trainer, batchSize, "measured", 6);
        Arrays.sort(seconds);
        double median = (seconds[2] + seconds[3]) / 2;
        long parameters = model.parameters().stream().mapToLong(p -> p.value.data.length).sum();
        System.out.printf("RESULT profile=%s parameters=%d batch=%d sequence=1024 medianSeconds=%.6f%n",
                profile, parameters, batchSize, median);
    }

    private static double[] rounds(Trainer trainer, int batchSize, String phase, int count) {
        double[] seconds = new double[count];
        for (int round = 0; round < count; round++) {
            long started = System.nanoTime();
            float loss = trainer.trainStep(batchSize);
            Tensor.backend().releaseTemporaryResources();
            seconds[round] = (System.nanoTime() - started) / 1_000_000_000.0;
            if (!Float.isFinite(loss)) throw new IllegalStateException("Non-finite benchmark loss");
            System.out.printf("phase=%s round=%d seconds=%.6f loss=%.7f%n", phase, round, seconds[round], loss);
        }
        return seconds;
    }

    private static Batch batch(int batchSize) {
        int[][] input = new int[batchSize][1_024], target = new int[batchSize][1_024];
        for (int row = 0; row < batchSize; row++) {
            for (int token = 0; token < 1_024; token++) {
                input[row][token] = (token * 37 + row * 5 + 11) % 16_384;
                target[row][token] = ((token + 1) * 37 + row * 5 + 11) % 16_384;
            }
        }
        return new Batch(input, target);
    }
}
