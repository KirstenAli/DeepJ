package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.training.TrainingResult;

import java.io.IOException;

public final class TrainDeepJPrismTinyStories {

    private TrainDeepJPrismTinyStories() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        run(DeepJPrismTinyStoriesConfig.fromSystemProperties());
    }

    public static TrainingResult run(DeepJPrismTinyStoriesConfig config) throws Exception {
        return DeepJPrismTrainingRunner.run(config);
    }

    public static TrainingResult runSequential(DeepJPrismTinyStoriesConfig config)
            throws Exception {
        return DeepJPrismTrainingRunner.runSequential(config);
    }

    static BPETokenizer prepareTokenizer(DeepJPrismTinyStoriesConfig config)
            throws IOException {
        return DeepJPrismTrainingRunner.prepareTokenizer(config);
    }
}
