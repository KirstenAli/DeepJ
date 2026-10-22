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
import java.util.Random;

/** Uses only APIs shared with the manual-backprop baseline; no dataset or checkpoint access. */
public final class AutogradTrainingBenchmark {
    private AutogradTrainingBenchmark() {}

    public static void main(String[] args) {
        if (!MetalBackend.isAvailable()) throw new IllegalStateException("This benchmark requires a Metal device");
        System.out.printf("START pid=%d%n", ProcessHandle.current().pid());
        Tensor.setBackend(new MetalBackend());
        try {
            benchmark(args[0], Integer.parseInt(args[1]), argument(args, 2, 6), argument(args, 3, 4));
        } finally {
            Tensor.backend().releaseResources();
        }
    }

    private static DecoderOnlyModel model(String profile) {
        return switch (profile) {
            case "prism", "prism-probe" -> new DeepJPrism(new DeepJPrismConfig(
                    16_384, 1_024, 768, 12, layers(profile, 10), 2_112, 384, 192, 0.2f, 1), 100);
            case "nova", "nova-probe" -> new DeepJNova(new DeepJNovaConfig(
                    16_384, 1_024, 576, 9, 3, layers(profile, 24), 1_664, true, 0.2f, 1), 100);
            default -> throw new IllegalArgumentException("Unknown benchmark profile: " + profile);
        };
    }

    private static int layers(String profile, int full) {
        return profile.endsWith("-probe") ? 2 : full;
    }

    private static int argument(String[] args, int index, int fallback) {
        return args.length > index ? Integer.parseInt(args[index]) : fallback;
    }

    private static void benchmark(String profile, int batchSize, int steps, int warmup) {
        if (batchSize < 1) throw new IllegalArgumentException("Batch size must be positive");
        if (steps < 1 || warmup < 1) throw new IllegalArgumentException("Round counts must be positive");
        long started = System.nanoTime();
        DecoderOnlyModel model = model(profile);
        Random random = Boolean.getBoolean("deepj.benchmarkVariedData") ? new Random(2_026) : null;
        Trainer trainer = CausalLMTraining.trainer(model, ignored -> batch(batchSize, random), 1e-4f);
        System.out.printf("CONFIG backend=%s profile=%s batch=%d steps=%d warmup=%d variedData=%s "
                        + "layers=%d initSeconds=%.6f%n",
                Tensor.backend().getClass().getSimpleName(), profile, batchSize, steps, warmup,
                random != null, model.config().nLayers(), (System.nanoTime() - started) / 1_000_000_000.0);
        rounds(trainer, batchSize, "warmup", warmup);
        try (var ignored = BenchmarkRecording.start()) {
            summarize(model, profile, batchSize, rounds(trainer, batchSize, "measured", steps));
        }
    }

    private static void summarize(DecoderOnlyModel model, String profile, int batchSize, double[] seconds) {
        Arrays.sort(seconds);
        double median = (seconds[(seconds.length - 1) / 2] + seconds[seconds.length / 2]) / 2;
        long parameters = model.parameters().stream().mapToLong(p -> p.value.data.length).sum();
        System.out.printf("RESULT profile=%s parameters=%d batch=%d sequence=1024 medianSeconds=%.6f%n",
                profile, parameters, batchSize, median);
    }

    private static double[] rounds(Trainer trainer, int batchSize, String phase, int count) {
        double[] seconds = new double[count];
        for (int round = 0; round < count; round++) {
            BenchmarkStep event = BenchmarkStep.start(phase, round);
            long started = System.nanoTime();
            float loss = trainer.trainStep(batchSize);
            Tensor.backend().releaseTemporaryResources();
            seconds[round] = (System.nanoTime() - started) / 1_000_000_000.0;
            event.finish();
            if (!Float.isFinite(loss)) throw new IllegalStateException("Non-finite benchmark loss");
            System.out.printf("phase=%s round=%d seconds=%.6f loss=%.7f %s%n",
                    phase, round, seconds[round], loss, BenchmarkTelemetry.snapshot());
        }
        return seconds;
    }

    private static Batch batch(int batchSize, Random random) {
        int[][] input = new int[batchSize][1_024], target = new int[batchSize][1_024];
        for (int row = 0; row < batchSize; row++) {
            int previous = token(random, row * 5 + 11);
            for (int token = 0; token < 1_024; token++) {
                input[row][token] = previous;
                previous = token(random, (token + 1) * 37 + row * 5 + 11);
                target[row][token] = previous;
            }
        }
        return new Batch(input, target);
    }

    private static int token(Random random, int offset) {
        return random == null ? offset % 16_384 : random.nextInt(16_384);
    }
}
