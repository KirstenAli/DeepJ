package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekParameterCount;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.nio.file.Path;

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
            DeepSeekTrainingRunner.prepareTokenizer(config);
            return;
        }
        DeepSeekTrainingRunner.runSequential(config);
    }

    static DeepSeekTinyStoriesConfig configuration() {
        Path output = path("deepj.output", "checkpoints/deepj-90m/pretrain");
        return new DeepSeekTinyStoriesConfig(files(output), architecture(), training(),
                tokenizer(), Long.getLong("deepj.seed", 90L));
    }

    static DeepSeekTinyStoriesConfig.Architecture architecture() {
        return new DeepSeekTinyStoriesConfig.Architecture(
                integer("deepj.seqLen", SEQUENCE_LENGTH), integer("deepj.dModel", D_MODEL),
                integer("deepj.heads", HEADS), integer("deepj.layers", LAYERS),
                integer("deepj.dFF", D_FF), integer("deepj.qRank", Q_RANK),
                integer("deepj.kvRank", KV_RANK), decimal("deepj.initScale", 0.2f),
                decimal("deepj.gradClipNorm", 1.0f));
    }

    private static DeepSeekTinyStoriesConfig.FilesConfig files(Path output) {
        Path corpus = path("deepj.corpus", "sample_data/deepj-90m/fineweb-edu.txt");
        String resume = System.getProperty("deepj.resume");
        return new DeepSeekTinyStoriesConfig.FilesConfig(
                corpus, output, resume == null ? null : Path.of(resume));
    }

    private static DeepSeekTinyStoriesConfig.Training training() {
        return new DeepSeekTinyStoriesConfig.Training(
                integer("deepj.steps", PRETRAINING_STEPS), integer("deepj.batchSize", 1),
                decimal("deepj.learningRate", 1e-4f), decimal("deepj.minLearningRate", 1e-5f),
                integer("deepj.warmupSteps", 2_000), integer("deepj.logEvery", 100),
                integer("deepj.checkpointEvery", 1_000), integer("deepj.releaseEvery", 1));
    }

    private static DeepSeekTinyStoriesConfig.TokenizerConfig tokenizer() {
        return new DeepSeekTinyStoriesConfig.TokenizerConfig(
                integer("deepj.vocabSize", VOCAB_SIZE),
                integer("deepj.tokenizerSampleMiB", 50));
    }

    private static void printRun(DeepSeekTinyStoriesConfig config) {
        long parameters = DeepSeekParameterCount.count(config.modelConfig(config.tokenizer().vocabSize()));
        System.out.printf("Backend: %s%nParameters: %,d%n", Tensor.backend().getClass().getSimpleName(),
                parameters);
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
