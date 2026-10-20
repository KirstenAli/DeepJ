package io.github.kirstenali.deepj.examples;

/** Fine-tunes Nova on Frisky's unchanged response-only dataset. */
public final class FineTuneDeepJ100M {

    private FineTuneDeepJ100M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        DeepJNovaTrainingRunner.run(DeepJ100MConfig.fromSystemProperties(DeepJ100MStage.FINE_TUNE));
    }
}
