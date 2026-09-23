package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

public final class CausalLMTestSupport {

    private CausalLMTestSupport() {}

    public static void assertLogitShape(CausalLM model, TransformerConfig config, int[] tokens) {
        Tensor logits = model.forward(tokens);
        assertEquals(tokens.length, logits.rows);
        assertEquals(config.vocabSize(), logits.cols);
    }

    public static void assertFullContext(CausalLM model, TransformerConfig config) {
        assertDoesNotThrow(() -> model.forward(new int[config.maxSeqLen()]));
    }

    public static void assertBackward(CausalLM model) {
        Tensor logits = model.forward(new int[]{ 1, 2, 3 });
        model.zeroGrad();
        model.backward(Tensor.ones(logits.rows, logits.cols));
        assertTrue(model.parameters().stream().anyMatch(parameter -> parameter.grad.sumAbs() > 0.0f));
    }

    public static void assertParameterCount(CausalLM model, TransformerConfig config, int perBlock) {
        int expected = 1 + config.nLayers() * perBlock + 1 + 2;
        assertEquals(expected, model.parameters().size());
    }

    public static void assertTrainingConfig(CausalLM model, TransformerConfig expected,
                                            TransformerConfig actual) {
        assertEquals(expected.gradClipNorm(), model.gradClipNorm());
        assertSame(expected, actual);
    }

    public static void assertCheckpoint(DecoderOnlyModel model, DecoderOnlyModel restored,
                                        Path checkpoint) throws IOException {
        float[] expected = materialized(model.forward(new int[]{ 1, 2, 3 }));
        model.save(checkpoint);
        restored.load(checkpoint);
        assertArrayEquals(expected, materialized(restored.forward(new int[]{ 1, 2, 3 })));
    }

    public static void assertGeneration(CausalLM model, TransformerConfig config) {
        var tokenizer = new ByteTokenizer();
        String output = TextGenerator.generate(model::forward, config, tokenizer, "hi", 8, 1.0f, 0, 1L);
        assertNotNull(output);
        assertTrue(output.startsWith("hi"));
    }

    public static void assertRepeatableGeneration(CausalLM model, TransformerConfig config) {
        var tokenizer = new ByteTokenizer();
        String first = TextGenerator.generate(model::forward, config, tokenizer, "hello", 10, 0.8f, 5, 42L);
        String second = TextGenerator.generate(model::forward, config, tokenizer, "hello", 10, 0.8f, 5, 42L);
        assertEquals(first, second);
    }

    public static boolean isAllOnes(Tensor tensor) {
        for (float value : tensor.data) {
            if (value != 1.0f) return false;
        }
        return true;
    }

    private static float[] materialized(Tensor tensor) {
        tensor.materialize();
        return tensor.data.clone();
    }
}
