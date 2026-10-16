package io.github.kirstenali.deepj.examples;

import java.util.Map;

final class TrainingMetadata {

    private TrainingMetadata() {}

    static void add(Map<String, String> target, Settings training) {
        target.put("steps", Integer.toString(training.steps()));
        target.put("batchSize", Integer.toString(training.batchSize()));
        target.put("peakLearningRate", Float.toString(training.peakLearningRate()));
        target.put("minimumLearningRate", Float.toString(training.minimumLearningRate()));
        target.put("warmupSteps", Integer.toString(training.warmupSteps()));
        target.put("checkpointEvery", Integer.toString(training.checkpointEvery()));
        target.put("releaseEvery", Integer.toString(training.releaseEvery()));
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
