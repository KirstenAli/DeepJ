package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Runs the fine_tune stage using the existing Frisky dataset. */
public final class FineTuneDeepJLoop100M {

    private FineTuneDeepJLoop100M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        try {
            StagedDecoderTrainingRunner.run(DeepJLoop100MConfig.fromSystemProperties(DeepJ100MStage.FINE_TUNE));
        } finally {
            Tensor.backend().releaseResources();
        }
    }
}
