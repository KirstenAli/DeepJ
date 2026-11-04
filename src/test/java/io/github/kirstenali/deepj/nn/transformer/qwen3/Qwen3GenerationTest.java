package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3GenerationTest {
    @ParameterizedTest @EnumSource(ContextWindowMode.class)
    void decodingAfterALongPromptReusesThePrefilledCache(ContextWindowMode mode) {
        var offsets = new ArrayList<Integer>();
        Tensor.setBackend(recordCacheWrites(offsets));
        var config = new Qwen3Config(19, 300, 12, 4, 2, 1, 20, 4, 1_000_000, 0.02f, 1);
        Qwen3 model = new Qwen3(config, 8);
        try (var session = model.openGeneration(mode)) {
            session.apply(new int[200]);
            for (int length = 201; length <= 280; length++) {
                session.apply(new int[length]);
            }
            assertEquals(2, offsets.stream().filter(offset -> offset == 0).count(),
                    "Only the initial key and value writes should start at position zero");
        } finally {
            cpu();
        }
    }

    private static TensorBackend recordCacheWrites(List<Integer> offsets) {
        var cpu = new CpuBackend();
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("writeHeadCache")) offsets.add((Integer) args[3]);
                    try { return method.invoke(cpu, args); }
                    catch (InvocationTargetException error) { throw error.getCause(); }
                });
    }

    @Test void cacheGrowthAndPromptResetsPreserveTheFullContextNumerics() {
        Tensor.setBackend(new CpuBackend());
        var config = new Qwen3Config(19, 260, 12, 4, 2, 1, 20, 4, 1_000_000, 0.02f, 1);
        Qwen3 model = new Qwen3(config, 8);
        int[] tokens = java.util.stream.IntStream.range(0, 259).map(index -> index % 19).toArray();
        try (var ignored = GradMode.noGrad(); var session = model.openGeneration()) {
            for (int length = 127; length <= 258; length++) {
                int[] prefix = java.util.Arrays.copyOf(tokens, length);
                float[] cached = session.apply(prefix).toFloatArray();
                if (java.util.Set.of(127, 128, 129, 130, 255, 256, 257, 258).contains(length)) {
                    assertArrayEquals(model.forward(prefix).select(0, length - 1).toFloatArray(), cached, 2e-6f);
                }
            }
            assertArrayEquals(model.forward(new int[]{4, 5}).select(0, 1).toFloatArray(),
                    session.apply(new int[]{4, 5}).toFloatArray(), 2e-6f);
        }
    }
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void cachedPrefillDecodeAndNewPromptsMatchFullForward(DType dtype) {
        Qwen3 model = new Qwen3(Qwen3ModelTest.tiny(), 8, dtype);
        int[] tokens = {1, 2, 3, 4, 5, 6};
        try (var session = model.openGeneration()) {
            for (int length = 2; length <= tokens.length; length++) {
                int[] prefix = Arrays.copyOf(tokens, length);
                assertArrayEquals(last(model.forward(prefix)), session.apply(prefix).toFloatArray(), 2e-6f);
            }
            int[] different = {7, 8, 9};
            assertArrayEquals(last(model.forward(different)), session.apply(different).toFloatArray(), 2e-6f);
        }
    }

    @Test void rollingCacheMatchesAnIndependentSlidingAttentionReferenceWithAbsolutePositions() {
        var config = new Qwen3Config(19, 4, 12, 4, 2, 2, 20, 4, 1_000_000, 0.02f, 1);
        Qwen3 model = new Qwen3(config, 11);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        var reference = new Qwen3Reference(model);
        try (var session = model.openGeneration(ContextWindowMode.ROLLING)) {
            for (int length = 2; length <= tokens.length; length++) {
                float[] expected = reference.logits(Arrays.copyOf(tokens, length), config.maxSeqLen());
                int[] window = Arrays.copyOfRange(tokens, Math.max(0, length - config.maxSeqLen()), length);
                assertArrayEquals(Arrays.copyOfRange(expected, expected.length - 19, expected.length),
                        session.apply(window).toFloatArray(), 2e-6f);
            }
        }
    }

    @Test void sessionsKeepIndependentCachesAndDetectChangedWeightsAndClosedSessions() {
        Qwen3 model = new Qwen3(Qwen3ModelTest.tiny(), 12);
        try (var first = model.openGeneration(); var second = model.openGeneration()) {
            first.apply(new int[]{1, 2});
            second.apply(new int[]{5, 6});
            assertArrayEquals(last(model.forward(new int[]{1, 2, 3})),
                    first.apply(new int[]{1, 2, 3}).toFloatArray(), 2e-6f);
            try (var ignored = GradMode.noGrad()) { model.parameters().get(0).value.addScalarInPlace(0.1f); }
            assertThrows(IllegalStateException.class, () -> second.apply(new int[]{5, 6, 7}));
        }
        var closed = model.openGeneration();
        closed.close();
        assertThrows(IllegalStateException.class, () -> closed.apply(new int[]{1}));
    }

    private static float[] last(Tensor logits) { return logits.select(0, logits.size(0) - 1).toFloatArray(); }
}
