package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.IOException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3HuggingFaceTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @ValueSource(strings = {"float32", "float16", "bfloat16"})
    void importedWeightsMatchOfficialTransformersLogitsAndEveryGradient(String dtype) throws IOException {
        Qwen3 model = Qwen3Pretrained.load(Qwen3Fixtures.path(dtype));
        var expected = Qwen3Fixtures.expected(dtype);
        int[][] tokens = Qwen3Fixtures.tokens(expected);
        Tensor logits = model.forwardBatch(tokens);
        assertArrayEquals(Qwen3Fixtures.floats(expected.get("logits")), logits.toFloatArray(), 2e-6f);
        logits.backward(Tensor.from(Qwen3Fixtures.floats(expected.get("upstream")), 2, 4, 19));
        var weights = Qwen3Weights.inventory(model.config()).stream()
                .collect(java.util.stream.Collectors.toMap(Qwen3Weights.Weight::name, weight -> weight));
        for (var parameter : model.parameters()) {
            assertEquals(DType.FLOAT32, parameter.grad().dtype());
            assertArrayEquals(Qwen3Fixtures.gradient(weights.get(parameter.name()), expected),
                    parameter.grad().toFloatArray(), 2e-5f, parameter.name());
        }
        assertEquals("4.51.3", Qwen3Json.string(expected.get("transformers")));
    }

    @ParameterizedTest @ValueSource(strings = {"float32", "float16", "bfloat16"})
    void cachedLogitsAndExplicitFloat32ImportMatchTheSameReference(String dtype) throws IOException {
        Qwen3 model = Qwen3Pretrained.load(Qwen3Fixtures.path(dtype), DType.FLOAT32);
        int[] tokens = Qwen3Fixtures.tokens(Qwen3Fixtures.expected(dtype))[0];
        try (var ignored = GradMode.noGrad(); var session = model.openGeneration()) {
            for (int length = 1; length <= tokens.length; length++) {
                int[] prefix = java.util.Arrays.copyOf(tokens, length);
                assertArrayEquals(model.forward(prefix).select(0, length - 1).toFloatArray(),
                        session.apply(prefix).toFloatArray(), 2e-6f);
            }
        }
        assertTrue(model.parameters().stream().allMatch(parameter -> parameter.value.dtype() == DType.FLOAT32));
    }
}
