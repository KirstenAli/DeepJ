package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.training.TrainingResult;

import java.io.IOException;

public final class TrainDeepSeekTinyStories {

    private TrainDeepSeekTinyStories() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        run(DeepSeekTinyStoriesConfig.fromSystemProperties());
    }

    public static TrainingResult run(DeepSeekTinyStoriesConfig config) throws Exception {
        return DeepSeekTrainingRunner.run(config);
    }

    public static TrainingResult runSequential(DeepSeekTinyStoriesConfig config)
            throws Exception {
        return DeepSeekTrainingRunner.runSequential(config);
    }

    static BPETokenizer prepareTokenizer(DeepSeekTinyStoriesConfig config)
            throws IOException {
        return DeepSeekTrainingRunner.prepareTokenizer(config);
    }
}
