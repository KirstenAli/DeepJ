package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.decimal;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.integer;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.path;

public final class MidTrainDeepJ90M {

    static final int MID_TRAINING_STEPS = 256_000;

    private MidTrainDeepJ90M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        var config = configuration();
        prepareTokenizer(config.files().outputDirectory());
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        DeepJPrismTrainingRunner.run(config);
    }

    static DeepJPrismTinyStoriesConfig configuration() {
        Path output = path("deepj.output", "checkpoints/deepj-90m/midtrain");
        var files = new DeepJPrismTinyStoriesConfig.FilesConfig(
                path("deepj.corpus", "sample_data/deepj-90m/midtrain-train.txt"), output, resume());
        return new DeepJPrismTinyStoriesConfig(files, TrainDeepJ90M.architecture(), training(),
                tokenizer(), Long.getLong("deepj.seed", 91L));
    }

    private static Path resume() {
        return path("deepj.resume", "checkpoints/deepj-90m/pretrain/model-final.dj");
    }

    private static DeepJPrismTinyStoriesConfig.Training training() {
        return new DeepJPrismTinyStoriesConfig.Training(
                integer("deepj.steps", MID_TRAINING_STEPS), integer("deepj.batchSize", 1),
                decimal("deepj.learningRate", 5e-5f), decimal("deepj.minLearningRate", 5e-6f),
                integer("deepj.warmupSteps", 500), integer("deepj.logEvery", 100),
                integer("deepj.checkpointEvery", 1_000), integer("deepj.releaseEvery", 1));
    }

    private static DeepJPrismTinyStoriesConfig.TokenizerConfig tokenizer() {
        return new DeepJPrismTinyStoriesConfig.TokenizerConfig(TrainDeepJ90M.VOCAB_SIZE, 50);
    }

    private static void prepareTokenizer(Path output) throws Exception {
        Files.createDirectories(output);
        Path source = path("deepj.base", "checkpoints/deepj-90m/pretrain").resolve("tokenizer.bpe");
        Files.copy(source, output.resolve("tokenizer.bpe"), StandardCopyOption.REPLACE_EXISTING);
    }

}
