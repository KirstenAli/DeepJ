package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.training.TrainingResult;

final class DeepJNovaTrainingRunner {

    private DeepJNovaTrainingRunner() {}

    static TrainingResult run(DeepJ100MConfig config) throws Exception {
        return StagedDecoderTrainingRunner.run(config);
    }
}
