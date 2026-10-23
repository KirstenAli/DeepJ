package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Runs the midtrain stage using the existing Frisky dataset. */
public final class MidTrainDeepJLoop100M {

    private MidTrainDeepJLoop100M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        try {
            StagedDecoderTrainingRunner.run(DeepJLoop100MConfig.fromSystemProperties(DeepJ100MStage.MIDTRAIN));
        } finally {
            Tensor.backend().releaseResources();
        }
    }
}
