package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.training.Trainer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Map;

/** Optional FP32 complete-step benchmark; not executed by the unit-test suite. */
public final class Qwen3TrainingBenchmark {
    private Qwen3TrainingBenchmark() {}

    public static void main(String[] args) throws Exception {
        boolean metal = args[0].equals("metal");
        if (metal && !MetalBackend.isAvailable()) throw new IllegalStateException("Physical Metal required");
        Tensor.setBackend(metal ? new MetalBackend() : new CpuBackend());
        DeepJExecutor.setNumThreads(10);
        try { run(args); }
        finally {
            Tensor.backend().releaseTemporaryResources();
            DeepJExecutor.shutdown();
        }
    }

    private static void run(String[] args) throws Exception {
        Qwen3 model = Qwen3TrainingFixture.model();
        Path output = Path.of(args[2]);
        if (args[1].equals("fixture")) {
            Qwen3TrainingFixture.save(model, output);
            return;
        }
        int warmups = args[1].equals("audit") ? 0 : 100;
        int samples = args[1].equals("audit") ? 1 : 30;
        if (args[1].equals("audit")) {
            Qwen3TrainingAudit.save(model, output.resolveSibling(output.getFileName() + "-raw"));
        }
        measure(model, output, args[0], warmups, samples);
    }

    private static void measure(Qwen3 model, Path output, String backend, int warmups, int samples) throws Exception {
        int[] tokens = Qwen3TrainingFixture.tokens(3), targets = Qwen3TrainingFixture.tokens(4);
        var trainer = CausalLMTraining.trainer(model, n -> new Batch(new int[][]{tokens}, new int[][]{targets}), 1e-4f);
        var times = new ArrayList<Double>();
        var losses = new ArrayList<Float>();
        for (int run = -warmups; run < samples; run++) {
            Step step = step(trainer);
            losses.add(step.loss());
            if (run >= 0) times.add(step.millis());
        }
        Qwen3Checkpoint.save(model, output);
        String json = Qwen3Json.stringify(Map.of("engine", "DeepJ", "backend", backend, "dtype", "FLOAT32",
                "warmups", warmups, "samples", samples, "ms", times, "losses", losses));
        Files.writeString(output.resolve("result.json"), json);
        System.out.println(json);
    }

    private static Step step(Trainer trainer) {
        long start = System.nanoTime();
        float loss = trainer.trainStep(1);
        Tensor.backend().releaseTemporaryResources();
        double millis = (System.nanoTime() - start) / 1e6;
        if (!Float.isFinite(loss)) throw new AssertionError("Non-finite loss");
        return new Step(loss, millis);
    }

    private record Step(float loss, double millis) {}
}
