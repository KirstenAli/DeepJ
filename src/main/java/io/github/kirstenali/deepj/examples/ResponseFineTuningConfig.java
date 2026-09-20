package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;

import java.nio.file.Path;
import java.util.List;

record ResponseFineTuningConfig(FilesConfig files, Training training,
                                List<ResponseOnlyTextDataset.Source> sources, long seed) {

    ResponseFineTuningConfig {
        if (files == null || training == null) throw new IllegalArgumentException("configuration is incomplete");
        if (sources == null || sources.isEmpty()) throw new IllegalArgumentException("sources are required");
        sources = List.copyOf(sources);
    }

    record FilesConfig(Path base, Path output, Path initialModel, Path resume) {

        FilesConfig {
            if (base == null || output == null || initialModel == null) {
                throw new IllegalArgumentException("file paths are required");
            }
        }
    }

    record Training(int steps, int batchSize, float peakLearningRate,
                    float minimumLearningRate, int warmupSteps, int logEvery,
                    int checkpointEvery, int releaseEvery,
                    int gradientAccumulationSteps) {

        Training(int steps, int batchSize, float peakLearningRate,
                 float minimumLearningRate, int warmupSteps, int logEvery,
                 int checkpointEvery, int releaseEvery) {
            this(steps, batchSize, peakLearningRate, minimumLearningRate, warmupSteps,
                    logEvery, checkpointEvery, releaseEvery, 1);
        }

        Training {
            if (steps < 1 || batchSize < 1 || logEvery < 1
                    || gradientAccumulationSteps < 1) {
                throw new IllegalArgumentException("training counts must be positive");
            }
            if (checkpointEvery < 0 || releaseEvery < 0) {
                throw new IllegalArgumentException("intervals must be non-negative");
            }
            new CosineLearningRateSchedule(peakLearningRate, minimumLearningRate,
                    warmupSteps, steps);
        }

        int effectiveBatchSize() {
            return Math.multiplyExact(batchSize, gradientAccumulationSteps);
        }
    }
}
