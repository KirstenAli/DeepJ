package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.nio.file.Path;
import java.util.List;

public final class FineTuneDeepJ90M {

    static final int FINE_TUNING_STEPS = 22_500;

    private FineTuneDeepJ90M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        ResponseOnlyFineTuner.run(configuration());
    }

    static ResponseFineTuningConfig configuration() {
        Path base = path("deepj.base", "checkpoints/deepj-90m/midtrain");
        var files = new ResponseFineTuningConfig.FilesConfig(base,
                path("deepj.output", "checkpoints/deepj-90m/sft"),
                path("deepj.initialModel", base.resolve("model-final.dj").toString()),
                optionalPath("deepj.resume"));
        var source = new ResponseOnlyTextDataset.Source(
                path("deepj.corpus", "sample_data/deepj-90m/sft-train.txt"), 1);
        return new ResponseFineTuningConfig(files, training(), List.of(source),
                Long.getLong("deepj.seed", 92L));
    }

    private static ResponseFineTuningConfig.Training training() {
        return new ResponseFineTuningConfig.Training(
                integer("deepj.steps", FINE_TUNING_STEPS), integer("deepj.batchSize", 1),
                decimal("deepj.learningRate", 2e-5f), decimal("deepj.minLearningRate", 2e-6f),
                integer("deepj.warmupSteps", 500), integer("deepj.logEvery", 100),
                integer("deepj.checkpointEvery", 1_000), integer("deepj.releaseEvery", 1));
    }

    private static int integer(String name, int fallback) {
        return Integer.parseInt(System.getProperty(name, Integer.toString(fallback)));
    }

    private static float decimal(String name, float fallback) {
        return Float.parseFloat(System.getProperty(name, Float.toString(fallback)));
    }

    private static Path path(String name, String fallback) {
        return Path.of(System.getProperty(name, fallback));
    }

    private static Path optionalPath(String name) {
        String value = System.getProperty(name);
        return value == null || value.isBlank() ? null : Path.of(value);
    }
}
