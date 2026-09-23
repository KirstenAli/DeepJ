package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.CausalLMEvaluation;

import java.nio.file.Path;
import java.util.List;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.decimal;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.integer;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.path;

public final class EvaluateDeepJ90M {

    private static final String[] PROMPTS = {
            "Hello",
            "Why is the sky blue?",
            "What does DNA stand for?",
            "What is 17 plus 28?",
            "Write a short story about a kind fox."
    };

    private EvaluateDeepJ90M() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        Settings settings = Settings.fromSystemProperties();
        var artifacts = DeepSeekTinyStoriesArtifacts.load(settings.base(), settings.checkpoint());
        evaluate(artifacts, settings);
        generate(artifacts, settings);
        Tensor.backend().releaseResources();
    }

    private static void evaluate(DeepSeekTinyStoriesArtifacts.Loaded artifacts,
                                 Settings settings) throws Exception {
        var source = new ResponseOnlyTextDataset.Source(settings.validation(), 1);
        var dataset = new ResponseOnlyTextDataset(List.of(source), artifacts.tokenizer(),
                artifacts.config().maxSeqLen(), settings.seed());
        var result = CausalLMEvaluation.evaluate(artifacts.model(), dataset,
                settings.batches(), 1);
        System.out.printf("Backend: %s%nValidation: tokens=%d loss=%.6f perplexity=%.3f%n",
                Tensor.backend().getClass().getSimpleName(), result.tokens(),
                result.loss(), result.perplexity());
    }

    private static void generate(DeepSeekTinyStoriesArtifacts.Loaded artifacts,
                                 Settings settings) {
        for (String prompt : PROMPTS) {
            String formatted = "Instruction:\n" + prompt + "\nResponse:\n";
            String text = TextGenerator.generate(artifacts.model(), artifacts.tokenizer(),
                    artifacts.config(), formatted, settings.generateTokens(),
                    settings.temperature(), settings.topK(), settings.seed());
            System.out.println("\n=== Sample ===\n" + text);
        }
    }

    private record Settings(Path base, Path checkpoint, Path validation, int batches,
                            int generateTokens, float temperature, int topK, long seed) {

        private static Settings fromSystemProperties() {
            Path base = path("deepj.base", "checkpoints/deepj-90m/sft");
            return new Settings(base,
                    path("deepj.checkpoint", base.resolve("model-final.dj").toString()),
                    path("deepj.validation", "sample_data/deepj-90m/sft-valid.txt"),
                    integer("deepj.evalBatches", 100), integer("deepj.generateTokens", 80),
                    decimal("deepj.temperature", 0.2f), integer("deepj.topK", 20),
                    Long.getLong("deepj.evalSeed", 1_000_090L));
        }

    }
}
