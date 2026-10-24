package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Runs the pretrain stage using the existing Frisky dataset. */
public final class TrainDeepJLoop100M {

    private TrainDeepJLoop100M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        try {
            StagedDecoderTrainingRunner.run(DeepJLoop100MConfig.fromSystemProperties(DeepJ100MStage.PRETRAIN));
        } finally {
            Tensor.backend().releaseResources();
        }
    }
}
