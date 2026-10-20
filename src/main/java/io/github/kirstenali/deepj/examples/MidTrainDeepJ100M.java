package io.github.kirstenali.deepj.examples;

/** Continues Nova on Frisky's unchanged broad instruction mixture. */
public final class MidTrainDeepJ100M {

    private MidTrainDeepJ100M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        DeepJNovaTrainingRunner.run(DeepJ100MConfig.fromSystemProperties(DeepJ100MStage.MIDTRAIN));
    }
}
