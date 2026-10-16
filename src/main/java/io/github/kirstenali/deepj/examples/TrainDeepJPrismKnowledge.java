package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.nio.file.Path;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.decimal;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.integer;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.nullablePath;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.path;

public final class TrainDeepJPrismKnowledge {

    static final int VOCAB_SIZE = 8_192;
    static final int SEQUENCE_LENGTH = 512;
    static final int D_MODEL = 384;
    static final int HEADS = 6;
    static final int LAYERS = 8;
    static final int D_FF = 1_024;
    static final int Q_RANK = 192;
    static final int KV_RANK = 96;
    static final int RELEASE_EVERY = 25;

    private TrainDeepJPrismKnowledge() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        DeepJPrismTrainingRunner.run(configuration());
    }

    static DeepJPrismTrainingConfig configuration() {
        Path output = path("deepj.output", "checkpoints/knowledge-prism");
        return new DeepJPrismTrainingConfig(
                files(output), architecture(), training(), tokenizer(),
                Long.getLong("deepj.seed", 42L));
    }

    private static DeepJPrismTrainingConfig.FilesConfig files(Path output) {
        Path corpus = path("deepj.corpus", output.resolve("knowledge-train.txt").toString());
        return new DeepJPrismTrainingConfig.FilesConfig(corpus, output, nullablePath("deepj.resume"));
    }

    private static DeepJPrismTrainingConfig.Architecture architecture() {
        return new DeepJPrismTrainingConfig.Architecture(
                integer("deepj.seqLen", SEQUENCE_LENGTH), integer("deepj.dModel", D_MODEL),
                integer("deepj.heads", HEADS), integer("deepj.layers", LAYERS),
                integer("deepj.dFF", D_FF), integer("deepj.qRank", Q_RANK),
                integer("deepj.kvRank", KV_RANK), decimal("deepj.initScale", 0.2f),
                decimal("deepj.gradClipNorm", 1.0f));
    }

    private static DeepJPrismTrainingConfig.Training training() {
        return new DeepJPrismTrainingConfig.Training(
                integer("deepj.steps", 100_000), integer("deepj.batchSize", 1),
                decimal("deepj.learningRate", 2e-4f),
                decimal("deepj.minLearningRate", 2e-5f),
                integer("deepj.warmupSteps", 2_000), integer("deepj.logEvery", 100),
                integer("deepj.checkpointEvery", 1_000),
                integer("deepj.releaseEvery", RELEASE_EVERY));
    }

    private static DeepJPrismTrainingConfig.TokenizerConfig tokenizer() {
        return new DeepJPrismTrainingConfig.TokenizerConfig(
                integer("deepj.vocabSize", VOCAB_SIZE),
                integer("deepj.tokenizerSampleMiB", 16));
    }

}
