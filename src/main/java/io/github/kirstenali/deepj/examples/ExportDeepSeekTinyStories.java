package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.publishing.DeepJModelBundle;
import io.github.kirstenali.deepj.publishing.ModelCard;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.training.CausalLMEvaluation;
import io.github.kirstenali.deepj.training.EvaluationResult;

import java.nio.file.Path;
import java.util.Properties;

/** Evaluates and exports the trained TinyStories checkpoint for Hugging Face. */
public final class ExportDeepSeekTinyStories {

    private static final String DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories";

    private ExportDeepSeekTinyStories() {}

    public static void main(String[] args) throws Exception {
        Path output = Path.of(System.getProperty("deepj.output", "checkpoints/tinystories-deepseek"));
        Path checkpoint = output.resolve("model-final.dj");
        Path bundle = Path.of(System.getProperty("deepj.bundle", "dist/deepj-tinystories-deepseek"));
        Path validation = Path.of(System.getProperty(
                "deepj.validationCorpus", "sample_data/TinyStories-valid.txt"));
        export(output, checkpoint, bundle, validation,
                Integer.getInteger("deepj.evalBatches", 100));
    }

    static Path export(Path output, Path checkpoint, Path bundle, Path validation,
                       int evaluationBatches) throws Exception {
        var artifacts = DeepSeekTinyStoriesArtifacts.load(output, checkpoint);
        EvaluationResult result = evaluate(artifacts, validation, evaluationBatches);
        ModelCard card = modelCard(artifacts, result, evaluationBatches);
        Path exported = DeepJModelBundle.export(bundle, artifacts.model(),
                artifacts.tokenizer().model(), card);
        verify(exported, artifacts);
        printResult(exported, result, parameterCount(artifacts));
        return exported;
    }

    private static void verify(Path bundle, DeepSeekTinyStoriesArtifacts.Loaded artifacts)
            throws Exception {
        BPEModel tokenizer = BPEModelIO.load(bundle.resolve(DeepJModelBundle.TOKENIZER_FILE));
        if (tokenizer.vocabSize() != artifacts.config().vocabSize()) {
            throw new IllegalStateException("Exported tokenizer vocabulary does not match the model");
        }
        long seed = DeepSeekTinyStoriesArtifacts.longValue(artifacts.properties(), "seed");
        DeepSeekModel reloaded = new DeepSeekModel(artifacts.config(), seed);
        reloaded.load(bundle.resolve(DeepJModelBundle.MODEL_FILE));
    }

    private static EvaluationResult evaluate(DeepSeekTinyStoriesArtifacts.Loaded artifacts,
                                             Path validation, int batches) throws Exception {
        long seed = Long.getLong("deepj.evalSeed", 1_000_045L);
        try (RandomAccessTextDataset source = new RandomAccessTextDataset(validation,
                artifacts.tokenizer(), artifacts.config().maxSeqLen(), seed)) {
            return CausalLMEvaluation.evaluate(artifacts.model(), source, batches, 1);
        }
    }

    private static ModelCard modelCard(DeepSeekTinyStoriesArtifacts.Loaded artifacts,
                                       EvaluationResult result, int batches) {
        long parameters = parameterCount(artifacts);
        String name = System.getProperty("deepj.modelName", "DeepJ TinyStories DeepSeek-style");
        String summary = summary(parameters, result, batches);
        String license = System.getProperty("deepj.modelLicense", "mit");
        return new ModelCard(name, summary, license, "en",
                trainingData(artifacts.properties()), limitations());
    }

    private static String summary(long parameters, EvaluationResult result, int batches) {
        return "An experimental %,d-parameter causal language model trained with DeepJ. "
                .formatted(parameters)
                + "On %d deterministic TinyStories validation windows (%d tokens), loss was %.6f "
                .formatted(batches, result.tokens(), result.loss())
                + "and perplexity was %.3f.".formatted(result.perplexity());
    }

    private static String trainingData(Properties properties) {
        return "Trained for %s steps with batch size %s and %s-token sequences on "
                .formatted(properties.getProperty("steps"), properties.getProperty("batchSize"),
                        properties.getProperty("sequenceLength"))
                + "[TinyStories](%s), a synthetic English dataset licensed CDLA-Sharing-1.0. "
                .formatted(DATASET_URL)
                + "The BPE vocabulary was trained from a bounded sample of the training split.";
    }

    private static String limitations() {
        return "Small experimental model trained only on synthetic children's stories. It may "
                + "produce incorrect, repetitive, biased, or unsuitable text. It has not been "
                + "evaluated for safety or downstream use and is not a general-purpose assistant.";
    }

    private static long parameterCount(DeepSeekTinyStoriesArtifacts.Loaded artifacts) {
        return artifacts.model().parameters().stream()
                .mapToLong(parameter -> parameter.value.data.length).sum();
    }

    private static void printResult(Path bundle, EvaluationResult result, long parameters) {
        System.out.printf("Exported %,d parameters to %s%n", parameters, bundle);
        System.out.printf("Validation: tokens=%d loss=%.6f perplexity=%.3f%n",
                result.tokens(), result.loss(), result.perplexity());
    }
}
