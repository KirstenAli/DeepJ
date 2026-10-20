package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.models.nova.DeepJNova;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.models.nova.DeepJNovaParameterCount;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.Trainer;

/** Measures synthetic full training steps without reading datasets or saving checkpoints. */
public final class BenchmarkDeepJ100M {

    private BenchmarkDeepJ100M() {}

    public static void main(String[] args) {
        if (!MetalBackend.isAvailable() && !Boolean.getBoolean("deepj.allowCpuBenchmark")) {
            throw new IllegalStateException("Metal is unavailable; run this benchmark in your Mac terminal");
        }
        TrainingExampleSupport.configureBackend();
        try {
            benchmark();
        } finally {
            Tensor.backend().releaseResources();
        }
    }

    private static void benchmark() {
        var config = DeepJ100MConfig.fromSystemProperties(DeepJ100MStage.PRETRAIN).architecture();
        long started = System.nanoTime();
        var model = new DeepJNova(config, 100L);
        Batch batch = batch(config);
        Trainer trainer = CausalLMTraining.trainer(model, ignored -> batch, 1e-4f);
        System.out.printf("backend=%s parameters=%,d sequenceLength=%d initSeconds=%.3f%n",
                Tensor.backend().getClass().getSimpleName(), parameterCount(model),
                config.maxSeqLen(), secondsSince(started));
        rounds(trainer, 2, "warmup");
        rounds(trainer, Integer.getInteger("deepj.benchmarkSteps", 3), "measured");
    }

    private static void rounds(Trainer trainer, int count, String label) {
        if (count < 1) throw new IllegalArgumentException("Benchmark steps must be positive");
        for (int round = 0; round < count; round++) {
            long started = System.nanoTime();
            float loss = trainer.trainStep(1);
            Tensor.backend().releaseTemporaryResources();
            System.out.printf("%s round=%d seconds=%.3f loss=%.6f%n",
                    label, round, secondsSince(started), loss);
        }
    }

    private static Batch batch(DeepJNovaConfig config) {
        int[] input = new int[config.maxSeqLen()];
        int[] target = new int[input.length];
        for (int index = 0; index < input.length; index++) {
            input[index] = (index * 37 + 11) % config.vocabSize();
            target[index] = ((index + 1) * 37 + 11) % config.vocabSize();
        }
        return new Batch(new int[][]{input}, new int[][]{target});
    }

    private static double secondsSince(long started) {
        return (System.nanoTime() - started) / 1_000_000_000.0;
    }

    private static long parameterCount(DeepJNova model) {
        long count = model.parameters().stream().mapToLong(parameter -> parameter.value.data.length).sum();
        if (count != DeepJNovaParameterCount.count(model.config())) {
            throw new IllegalStateException("Parameter count does not match the model configuration");
        }
        return count;
    }
}
