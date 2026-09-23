package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.prism.DeepJPrismParameterCount;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.nio.file.Path;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.decimal;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.integer;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.nullablePath;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.path;

public final class TrainDeepJ90M {

    static final int VOCAB_SIZE = 16_384;
    static final int SEQUENCE_LENGTH = 1_024;
    static final int D_MODEL = 768;
    static final int HEADS = 12;
    static final int LAYERS = 10;
    static final int D_FF = 2_112;
    static final int Q_RANK = 384;
    static final int KV_RANK = 192;
    static final int PRETRAINING_STEPS = 1_757_813;

    private TrainDeepJ90M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        var config = configuration();
        printRun(config);
        if (Boolean.getBoolean("deepj.tokenizerOnly")) {
            DeepJPrismTrainingRunner.prepareTokenizer(config);
            return;
        }
        DeepJPrismTrainingRunner.runSequential(config);
    }

    static DeepJPrismTinyStoriesConfig configuration() {
        Path output = path("deepj.output", "checkpoints/deepj-90m/pretrain");
        return new DeepJPrismTinyStoriesConfig(files(output), architecture(), training(),
                tokenizer(), Long.getLong("deepj.seed", 90L));
    }

    static DeepJPrismTinyStoriesConfig.Architecture architecture() {
        return new DeepJPrismTinyStoriesConfig.Architecture(
                integer("deepj.seqLen", SEQUENCE_LENGTH), integer("deepj.dModel", D_MODEL),
                integer("deepj.heads", HEADS), integer("deepj.layers", LAYERS),
                integer("deepj.dFF", D_FF), integer("deepj.qRank", Q_RANK),
                integer("deepj.kvRank", KV_RANK), decimal("deepj.initScale", 0.2f),
                decimal("deepj.gradClipNorm", 1.0f));
    }

    private static DeepJPrismTinyStoriesConfig.FilesConfig files(Path output) {
        Path corpus = path("deepj.corpus", "sample_data/deepj-90m/fineweb-edu.txt");
        return new DeepJPrismTinyStoriesConfig.FilesConfig(corpus, output, nullablePath("deepj.resume"));
    }

    private static DeepJPrismTinyStoriesConfig.Training training() {
        return new DeepJPrismTinyStoriesConfig.Training(
                integer("deepj.steps", PRETRAINING_STEPS), integer("deepj.batchSize", 1),
                decimal("deepj.learningRate", 1e-4f), decimal("deepj.minLearningRate", 1e-5f),
                integer("deepj.warmupSteps", 2_000), integer("deepj.logEvery", 100),
                integer("deepj.checkpointEvery", 1_000), integer("deepj.releaseEvery", 1));
    }

    private static DeepJPrismTinyStoriesConfig.TokenizerConfig tokenizer() {
        return new DeepJPrismTinyStoriesConfig.TokenizerConfig(
                integer("deepj.vocabSize", VOCAB_SIZE),
                integer("deepj.tokenizerSampleMiB", 50));
    }

    private static void printRun(DeepJPrismTinyStoriesConfig config) {
        long parameters = DeepJPrismParameterCount.count(config.modelConfig(config.tokenizer().vocabSize()));
        System.out.printf("Backend: %s%nParameters: %,d%n", Tensor.backend().getClass().getSimpleName(),
                parameters);
    }

}
