package io.github.kirstenali.deepj.examples;

/** Trains Nova from scratch using Frisky's FineWeb-Edu corpus and tokenizer. */
public final class TrainDeepJ100M {

    private TrainDeepJ100M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        DeepJNovaTrainingRunner.run(DeepJ100MConfig.fromSystemProperties(DeepJ100MStage.PRETRAIN));
    }
}
