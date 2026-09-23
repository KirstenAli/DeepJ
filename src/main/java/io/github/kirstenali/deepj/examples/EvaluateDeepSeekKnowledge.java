package io.github.kirstenali.deepj.examples;

import io.github.kirstenali.deepj.data.ResponseOnlyTextDataset;
import io.github.kirstenali.deepj.models.TextGenerator;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.training.CausalLMEvaluation;
import io.github.kirstenali.deepj.training.EvaluationResult;

import java.nio.file.Path;
import java.util.List;

import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.decimal;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.integer;
import static io.github.kirstenali.deepj.examples.ExampleSystemProperties.path;

public final class EvaluateDeepSeekKnowledge {

    private static final String[] SAMPLE_PROMPTS = {
            "What is 17 plus 28?",
            "What does DNA stand for?",
            "Why do plants need sunlight?",
            "Write a short story about a kind fox."
    };

    private EvaluateDeepSeekKnowledge() {}

    public static void main(String[] args) throws Exception {
        TrainingExampleSupport.configureBackend();
        Settings settings = Settings.fromSystemProperties();
        var artifacts = DeepSeekTinyStoriesArtifacts.load(settings.base(), settings.checkpoint());
        System.out.println("Backend: " + Tensor.backend().getClass().getSimpleName());
        evaluate(settings, artifacts);
        Tensor.backend().releaseResources();
    }

    private static void evaluate(Settings settings,
                                 DeepSeekTinyStoriesArtifacts.Loaded artifacts) throws Exception {
        var alpaca = dataset(settings.alpaca(), artifacts, settings.seed());
        var facts = dataset(settings.facts(), artifacts, settings.seed() + 1);
        print("Alpaca", evaluate(artifacts, alpaca, settings.batches()));
        print("Facts", evaluate(artifacts, facts, settings.batches()));
        printAccuracy(exactAccuracy(artifacts, facts, settings.exactSamples()));
        if (settings.generateSamples()) generateSamples(artifacts, settings);
    }

    private static ResponseOnlyTextDataset dataset(
            Path path, DeepSeekTinyStoriesArtifacts.Loaded artifacts, long seed) throws Exception {
        var source = new ResponseOnlyTextDataset.Source(path, 1);
        return new ResponseOnlyTextDataset(List.of(source), artifacts.tokenizer(),
                artifacts.config().maxSeqLen(), seed);
    }

    private static EvaluationResult evaluate(DeepSeekTinyStoriesArtifacts.Loaded artifacts,
                                             ResponseOnlyTextDataset dataset, int batches) {
        return CausalLMEvaluation.evaluate(artifacts.model(), dataset, batches, 1);
    }

    private static AccuracyResult exactAccuracy(DeepSeekTinyStoriesArtifacts.Loaded artifacts,
                                                ResponseOnlyTextDataset dataset, int requested) {
        List<ResponseOnlyTextDataset.Example> facts = arithmetic(dataset.examples());
        int samples = Math.min(requested, facts.size());
        int correct = 0;
        int shown = 0;
        for (int index = 0; index < samples; index++) {
            var example = facts.get(index * facts.size() / samples);
            String answer = greedyAnswer(artifacts, example.prompt());
            if (example.response().equals(answer)) correct++;
            else if (shown++ < 5) printMiss(example, answer);
        }
        return new AccuracyResult(correct, samples);
    }

    private static List<ResponseOnlyTextDataset.Example> arithmetic(
            List<ResponseOnlyTextDataset.Example> examples) {
        return examples.stream().filter(example -> example.response().matches("[0-9]+\\.")).toList();
    }

    private static String greedyAnswer(DeepSeekTinyStoriesArtifacts.Loaded artifacts,
                                       String prompt) {
        String generated = TextGenerator.generate(artifacts.model(), artifacts.tokenizer(),
                artifacts.config(), prompt, 12, 1.0f, 1, 2026L);
        return generated.substring(prompt.length()).trim();
    }

    private static void generateSamples(DeepSeekTinyStoriesArtifacts.Loaded artifacts,
                                        Settings settings) {
        for (String instruction : SAMPLE_PROMPTS) {
            String prompt = prompt(instruction);
            String output = TextGenerator.generate(artifacts.model(), artifacts.tokenizer(),
                    artifacts.config(), prompt, settings.generateTokens(),
                    settings.temperature(), settings.topK(), settings.seed());
            System.out.println("\n=== Sample ===\n" + output);
        }
    }

    private static String prompt(String instruction) {
        return "Instruction:\n" + instruction + "\nResponse:\n";
    }

    private static void print(String name, EvaluationResult result) {
        System.out.printf("%s validation: tokens=%d loss=%.6f perplexity=%.3f%n",
                name, result.tokens(), result.loss(), result.perplexity());
    }

    private static void printMiss(ResponseOnlyTextDataset.Example example, String actual) {
        System.out.printf("MISS %s expected=%s actual=%s%n", singleLine(example.prompt()),
                example.response(), singleLine(actual));
    }

    private static String singleLine(String value) {
        return value.replace('\n', ' ').strip();
    }

    private static void printAccuracy(AccuracyResult result) {
        System.out.printf("Exact arithmetic accuracy: %d/%d (%.1f%%)%n", result.correct(),
                result.total(), 100.0 * result.correct() / result.total());
    }

    private record AccuracyResult(int correct, int total) {}

    private record Settings(Path base, Path checkpoint, Path alpaca, Path facts,
                            int batches, int exactSamples, int generateTokens,
                            float temperature, int topK, long seed,
                            boolean generateSamples) {

        private static Settings fromSystemProperties() {
            Path base = path("deepj.base", "checkpoints/knowledge-deepseek");
            Path output = path("deepj.output", "checkpoints/knowledge-finetune");
            return new Settings(base, path("deepj.checkpoint", output.resolve("model-final.dj").toString()),
                    path("deepj.alpacaValidation", base.resolve("alpaca-valid.txt").toString()),
                    path("deepj.factValidation", base.resolve("facts-valid.txt").toString()),
                    integer("deepj.evalBatches", 100), integer("deepj.exactSamples", 100),
                    integer("deepj.generateTokens", 40), decimal("deepj.temperature", 0.1f),
                    integer("deepj.topK", 20), Long.getLong("deepj.evalSeed", 1_000_046L),
                    Boolean.parseBoolean(System.getProperty("deepj.generateSamples", "true")));
        }

    }
}
