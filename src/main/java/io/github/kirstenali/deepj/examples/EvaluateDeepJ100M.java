package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.nio.file.Path;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.integer;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.path;

/** Evaluates Nova on the same held-out FineWeb-Edu range used for Frisky. */
public final class EvaluateDeepJ100M {

    private EvaluateDeepJ100M() {}

    public static void main(String[] args) throws Exception {
        run("checkpoints/deepj-100m/pretrain");
    }

    static void run(String directory) throws Exception {
        TrainingExampleSupport.configureBackend();
        try {
            EvaluateDeepJ90MPretraining.run(settings(directory));
        } finally {
            Tensor.backend().releaseResources();
        }
    }

    private static EvaluateDeepJ90MPretraining.Settings settings(String directory) {
        Path base = path("deepj.base", directory);
        return new EvaluateDeepJ90MPretraining.Settings(base,
                path("deepj.checkpoint", base.resolve("model-latest.dj").toString()),
                path("deepj.corpus", "sample_data/deepj-90m/fineweb-edu.txt"),
                integer("deepj.evalBatches", 10), integer("deepj.evalBatchSize", 1),
                Long.getLong("deepj.evalSeed", 1_000_090L));
    }
}
