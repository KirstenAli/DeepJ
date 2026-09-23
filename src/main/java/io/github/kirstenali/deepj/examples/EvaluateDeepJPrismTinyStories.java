package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.CausalLMEvaluation;
import io.github.kirstenali.deepj.training.EvaluationResult;

import java.nio.file.Path;
import java.util.Properties;

public final class EvaluateDeepJPrismTinyStories {

    private EvaluateDeepJPrismTinyStories() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        Path output = Path.of(System.getProperty("deepj.output", "checkpoints/tinystories-prism"));
        var artifacts = DeepJPrismTinyStoriesArtifacts.load(output, checkpoint(output));
        EvaluationResult result = evaluate(artifacts);
        printResult(result);
        generate(artifacts);
    }

    private static EvaluationResult evaluate(DeepJPrismTinyStoriesArtifacts.Loaded artifacts)
            throws Exception {
        Properties properties = artifacts.properties();
        Path corpus = evaluationCorpus(properties);
        long seed = Long.getLong("deepj.evalSeed",
                DeepJPrismTinyStoriesArtifacts.longValue(properties, "seed") + 1_000_003L);
        int batches = Integer.getInteger("deepj.evalBatches", 100);
        int batchSize = Integer.getInteger("deepj.evalBatchSize", 1);
        try (RandomAccessTextDataset source =
                     new RandomAccessTextDataset(corpus, artifacts.tokenizer(),
                             artifacts.config().maxSeqLen(), seed)) {
            return CausalLMEvaluation.evaluate(artifacts.model(), source, batches, batchSize);
        }
    }

    private static Path evaluationCorpus(Properties properties) {
        String fallback = properties.getProperty("corpus");
        return Path.of(System.getProperty("deepj.evalCorpus", fallback));
    }

    private static void generate(DeepJPrismTinyStoriesArtifacts.Loaded artifacts) {
        String prompt = System.getProperty("deepj.prompt", "Once upon a time");
        String text = TextGenerator.generate(artifacts.model(), artifacts.tokenizer(),
                artifacts.config(), prompt,
                Integer.getInteger("deepj.generateTokens", 80), 0.8f, 40, 2026L);
        System.out.println("\n=== Sample ===\n" + text);
    }

    private static Path checkpoint(Path output) {
        return Path.of(System.getProperty("deepj.checkpoint",
                output.resolve("model-final.dj").toString()));
    }

    private static void printResult(EvaluationResult result) {
        System.out.printf("Backend: %s%nEvaluation: tokens=%d loss=%.6f perplexity=%.3f%n",
                Tensor.backend().getClass().getSimpleName(), result.tokens(),
                result.loss(), result.perplexity());
    }
}
