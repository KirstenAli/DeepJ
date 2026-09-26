package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.RandomAccessTextDataset;
import io.github.kirstenali.deepj.data.TextFileRange;
import io.github.kirstenali.deepj.models.CausalLM;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;

public final class FixedTextValidation {

    private FixedTextValidation() {}

    public static EvaluationResult evaluate(CausalLM model, Path corpus,
                                            Tokenizer tokenizer, Settings settings)
            throws IOException {
        return evaluate(model, corpus, tokenizer, settings, TextFileRange.entire(corpus));
    }

    public static EvaluationResult evaluate(CausalLM model, Path corpus,
                                            Tokenizer tokenizer, Settings settings,
                                            TextFileRange range) throws IOException {
        validate(model, corpus, tokenizer, settings);
        try (var dataset = new RandomAccessTextDataset(corpus, tokenizer,
                settings.sequenceLength(), settings.seed(), range)) {
            return CausalLMEvaluation.evaluate(model, dataset,
                    settings.batches(), settings.batchSize());
        }
    }

    private static void validate(CausalLM model, Path corpus, Tokenizer tokenizer,
                                 Settings settings) {
        if (model == null || corpus == null || tokenizer == null || settings == null) {
            throw new IllegalArgumentException("validation arguments must not be null");
        }
    }

    public record Settings(int sequenceLength, int batches, int batchSize, long seed) {

        public Settings {
            if (sequenceLength < 2 || batches <= 0 || batchSize <= 0) {
                throw new IllegalArgumentException("validation counts must be positive");
            }
        }
    }
}
