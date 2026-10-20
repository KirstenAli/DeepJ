package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.training.CosineLearningRateSchedule;

record TrainingOptions(int steps, int batchSize, float peakLearningRate,
                       float minimumLearningRate, int warmupSteps, int logEvery,
                       int checkpointEvery, int releaseEvery) implements TrainingMetadata.Settings {

    TrainingOptions {
        if (steps <= 0 || batchSize <= 0 || logEvery <= 0) {
            throw new IllegalArgumentException("steps, batchSize, and logEvery must be positive");
        }
        if (checkpointEvery < 0 || releaseEvery < 0) {
            throw new IllegalArgumentException("checkpointEvery and releaseEvery must be non-negative");
        }
        new CosineLearningRateSchedule(peakLearningRate, minimumLearningRate, warmupSteps, steps);
    }
}
