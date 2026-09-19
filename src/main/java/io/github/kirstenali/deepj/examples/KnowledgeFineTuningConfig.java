package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;

import java.nio.file.Path;
import java.util.List;

record KnowledgeFineTuningConfig(FilesConfig files, Training training,
                                 int alpacaWeight, int factWeight, long seed) {

    KnowledgeFineTuningConfig {
        if (files == null || training == null) throw new IllegalArgumentException("configuration is incomplete");
        if (alpacaWeight < 1 || factWeight < 1) throw new IllegalArgumentException("weights must be positive");
    }

    static KnowledgeFineTuningConfig fromSystemProperties() {
        Path base = path("deepj.base", "checkpoints/knowledge-deepseek");
        return new KnowledgeFineTuningConfig(FilesConfig.fromSystemProperties(base),
                Training.fromSystemProperties(), integer("deepj.alpacaWeight", 1),
                integer("deepj.factWeight", 1), Long.getLong("deepj.seed", 43L));
    }

    ResponseFineTuningConfig responseConfig() {
        var genericFiles = new ResponseFineTuningConfig.FilesConfig(
                files.base(), files.output(), files.initialModel(), files.resume());
        var sources = List.of(new ResponseOnlyTextDataset.Source(files.alpaca(), alpacaWeight),
                new ResponseOnlyTextDataset.Source(files.facts(), factWeight));
        return new ResponseFineTuningConfig(genericFiles, training.responseConfig(), sources, seed);
    }

    record FilesConfig(Path base, Path output, Path alpaca, Path facts,
                       Path initialModel, Path resume) {

        static FilesConfig fromSystemProperties(Path base) {
            return new FilesConfig(base, path("deepj.output", "checkpoints/knowledge-finetune"),
                    path("deepj.alpacaCorpus", base.resolve("alpaca-train.txt").toString()),
                    path("deepj.factCorpus", base.resolve("facts-train.txt").toString()),
                    path("deepj.initialModel", base.resolve("model-final.dj").toString()),
                    optionalPath("deepj.resume"));
        }
    }

    record Training(int steps, int batchSize, float peakLearningRate,
                    float minimumLearningRate, int warmupSteps, int logEvery,
                    int checkpointEvery, int releaseEvery) {

        Training {
            if (steps < 1 || batchSize < 1 || logEvery < 1) throw new IllegalArgumentException("training counts must be positive");
            if (checkpointEvery < 0 || releaseEvery < 0) throw new IllegalArgumentException("intervals must be non-negative");
            new CosineLearningRateSchedule(peakLearningRate, minimumLearningRate, warmupSteps, steps);
        }

        static Training fromSystemProperties() {
            return new Training(integer("deepj.steps", 20_000), integer("deepj.batchSize", 1),
                    decimal("deepj.learningRate", 5e-5f), decimal("deepj.minLearningRate", 5e-6f),
                    integer("deepj.warmupSteps", 500), integer("deepj.logEvery", 100),
                    integer("deepj.checkpointEvery", 1_000), integer("deepj.releaseEvery", 25));
        }

        ResponseFineTuningConfig.Training responseConfig() {
            return new ResponseFineTuningConfig.Training(steps, batchSize, peakLearningRate,
                    minimumLearningRate, warmupSteps, logEvery, checkpointEvery, releaseEvery);
        }
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
