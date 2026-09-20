package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.nio.file.Path;

public final class TrainDeepSeekKnowledge {

    static final int VOCAB_SIZE = 8_192;
    static final int SEQUENCE_LENGTH = 512;
    static final int D_MODEL = 384;
    static final int HEADS = 6;
    static final int LAYERS = 8;
    static final int D_FF = 1_024;
    static final int Q_RANK = 192;
    static final int KV_RANK = 96;
    static final int RELEASE_EVERY = 25;

    private TrainDeepSeekKnowledge() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        DeepSeekTrainingRunner.run(configuration());
    }

    static DeepSeekTinyStoriesConfig configuration() {
        Path output = path("deepj.output", "checkpoints/knowledge-deepseek");
        return new DeepSeekTinyStoriesConfig(
                files(output), architecture(), training(), tokenizer(),
                Long.getLong("deepj.seed", 42L));
    }

    private static DeepSeekTinyStoriesConfig.FilesConfig files(Path output) {
        Path corpus = path("deepj.corpus", output.resolve("knowledge-train.txt").toString());
        String resume = System.getProperty("deepj.resume");
        return new DeepSeekTinyStoriesConfig.FilesConfig(
                corpus, output, resume == null ? null : Path.of(resume));
    }

    private static DeepSeekTinyStoriesConfig.Architecture architecture() {
        return new DeepSeekTinyStoriesConfig.Architecture(
                integer("deepj.seqLen", SEQUENCE_LENGTH), integer("deepj.dModel", D_MODEL),
                integer("deepj.heads", HEADS), integer("deepj.layers", LAYERS),
                integer("deepj.dFF", D_FF), integer("deepj.qRank", Q_RANK),
                integer("deepj.kvRank", KV_RANK), decimal("deepj.initScale", 0.2f),
                decimal("deepj.gradClipNorm", 1.0f));
    }

    private static DeepSeekTinyStoriesConfig.Training training() {
        return new DeepSeekTinyStoriesConfig.Training(
                integer("deepj.steps", 100_000), integer("deepj.batchSize", 1),
                decimal("deepj.learningRate", 2e-4f),
                decimal("deepj.minLearningRate", 2e-5f),
                integer("deepj.warmupSteps", 2_000), integer("deepj.logEvery", 100),
                integer("deepj.checkpointEvery", 1_000),
                integer("deepj.releaseEvery", RELEASE_EVERY));
    }

    private static DeepSeekTinyStoriesConfig.TokenizerConfig tokenizer() {
        return new DeepSeekTinyStoriesConfig.TokenizerConfig(
                integer("deepj.vocabSize", VOCAB_SIZE),
                integer("deepj.tokenizerSampleMiB", 16));
    }

    private static int integer(String name, int fallback) {
        return Integer.parseInt(System.getProperty(name, Integer.toString(fallback)));
    }

    private static float decimal(String name, float fallback) {
        return Float.parseFloat(System.getProperty(name, Float.toString(fallback)));
    }

    private static Path path(String name, String fallback) {
        return Path.of(System.getProperty(name, fallback));
    }
}
