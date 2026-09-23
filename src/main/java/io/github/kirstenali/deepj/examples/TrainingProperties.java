package io.github.kirstenali.deepj.examples;

import java.util.Properties;

final class TrainingProperties {

    private TrainingProperties() {}

    static void add(Properties target, Settings training) {
        target.setProperty("steps", Integer.toString(training.steps()));
        target.setProperty("batchSize", Integer.toString(training.batchSize()));
        target.setProperty("peakLearningRate", Float.toString(training.peakLearningRate()));
        target.setProperty("minimumLearningRate", Float.toString(training.minimumLearningRate()));
        target.setProperty("warmupSteps", Integer.toString(training.warmupSteps()));
        target.setProperty("checkpointEvery", Integer.toString(training.checkpointEvery()));
        target.setProperty("releaseEvery", Integer.toString(training.releaseEvery()));
    }

    interface Settings {

        int steps();

        int batchSize();

        float peakLearningRate();

        float minimumLearningRate();

        int warmupSteps();

        int checkpointEvery();

        int releaseEvery();
    }
}
