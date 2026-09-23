package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOriginModel;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CausalLMEvaluationTest {

    @BeforeEach
    void useCpuBackend() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void evaluationReturnsFiniteTokenWeightedMetrics() throws Exception {
        Path corpus = Files.createTempFile("deepj-evaluation", ".txt");
        Files.writeString(corpus, "hello world ".repeat(20));
        TextDataset source = TextDataset.fromFile(corpus, new ByteTokenizer(), 8, 1L);
        DeepJOriginModel model = new DeepJOriginModel(new DeepJOriginConfig(256, 8, 8, 2, 1, 16), 2L);
        EvaluationResult result = CausalLMEvaluation.evaluate(model, source, 3, 2);
        assertTrue(Double.isFinite(result.loss()));
        assertEquals(Math.exp(result.loss()), result.perplexity(), 1e-12);
        assertEquals(48L, result.tokens());
    }

    @Test
    void evaluationRejectsInvalidCounts() {
        DeepJOriginModel model = new DeepJOriginModel(new DeepJOriginConfig(11, 4, 4, 2, 1, 8), 1L);
        assertThrows(IllegalArgumentException.class,
                () -> CausalLMEvaluation.evaluate(model, size -> null, 0, 1));
    }
}
