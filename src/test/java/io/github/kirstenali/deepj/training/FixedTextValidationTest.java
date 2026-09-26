package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FixedTextValidationTest {

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void repeatedEvaluationUsesIdenticalExamples() throws Exception {
        Path corpus = corpus();
        DeepJOrigin model = model();
        var settings = new FixedTextValidation.Settings(8, 3, 2, 19L);
        EvaluationResult first = evaluate(model, corpus, settings);
        EvaluationResult second = evaluate(model, corpus, settings);
        assertEquals(first, second);
        assertEquals(48L, first.tokens());
    }

    private static EvaluationResult evaluate(DeepJOrigin model, Path corpus,
                                             FixedTextValidation.Settings settings)
            throws Exception {
        return FixedTextValidation.evaluate(model, corpus, new ByteTokenizer(), settings);
    }

    private static DeepJOrigin model() {
        return new DeepJOrigin(new DeepJOriginConfig(256, 8, 8, 2, 1, 16), 2L);
    }

    private static Path corpus() throws Exception {
        Path path = Files.createTempFile("deepj-fixed-validation", ".txt");
        Files.writeString(path, "alpha beta gamma delta epsilon\n".repeat(400));
        return path;
    }
}
