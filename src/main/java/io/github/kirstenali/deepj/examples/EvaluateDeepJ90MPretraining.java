package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.EvaluationResult;
import io.github.kirstenali.deepj.training.FixedTextValidation;
import io.github.kirstenali.deepj.data.TextFileRange;

import java.nio.file.Path;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.integer;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.path;

public final class EvaluateDeepJ90MPretraining {

    private EvaluateDeepJ90MPretraining() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        try {
            run(Settings.fromSystemProperties());
        } finally {
            Tensor.backend().releaseResources();
        }
    }

    private static void run(Settings settings) throws Exception {
        var artifacts = DeepJPrismTinyStoriesArtifacts.load(
                settings.base(), settings.checkpoint());
        var split = DeepJ90MDataSplit.fromSystemProperties(settings.corpus());
        var validation = validationSettings(artifacts.config().maxSeqLen(), settings);
        var result = FixedTextValidation.evaluate(artifacts.model(),
                settings.corpus(), artifacts.tokenizer(), validation, split.validation());
        print(settings, split.validation(), result);
    }

    private static FixedTextValidation.Settings validationSettings(
            int sequenceLength, Settings settings) {
        return new FixedTextValidation.Settings(sequenceLength, settings.batches(),
                settings.batchSize(), settings.seed());
    }

    private static void print(Settings settings, TextFileRange range,
                              EvaluationResult result) {
        System.out.printf("Backend: %s%n", Tensor.backend().getClass().getSimpleName());
        System.out.printf("Validation: source=%s range=%d:%d seed=%d%n",
                settings.corpus(), range.startInclusive(), range.endExclusive(), settings.seed());
        System.out.printf("Validation: tokens=%d loss=%.6f perplexity=%.3f%n",
                result.tokens(), result.loss(), result.perplexity());
    }

    record Settings(Path base, Path checkpoint, Path corpus,
                    int batches, int batchSize, long seed) {

        static Settings fromSystemProperties() {
            Path base = path("deepj.base", "checkpoints/deepj-90m/pretrain");
            return new Settings(base,
                    path("deepj.checkpoint", base.resolve("model-latest.dj").toString()),
                    path("deepj.corpus", "sample_data/deepj-90m/fineweb-edu.txt"),
                    integer("deepj.evalBatches", 10), integer("deepj.evalBatchSize", 1),
                    Long.getLong("deepj.evalSeed", 1_000_090L));
        }
    }
}
