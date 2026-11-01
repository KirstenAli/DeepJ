package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.testing.InferenceTestModels;
import io.github.kirstenali.deepj.testing.InferenceTestModels.Kind;
import io.github.kirstenali.deepj.testing.SlidingWindowReference;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.Arguments;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class MetalRollingGenerationTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(value = Kind.class, names = "ORIGIN", mode = EnumSource.Mode.EXCLUDE)
    void rollingOptInPreservesEveryPreEvictionGpuLogitExactly(Kind kind) {
        DecoderOnlyModel model = InferenceTestModels.create(kind, 288, 128);
        int[] tokens = randomTargets(288, 31, 991);
        try (var exact = model.openGeneration(); var rolling = model.openGeneration(ContextWindowMode.ROLLING)) {
            for (int length = 280; length <= tokens.length; length++) {
                int[] prefix = Arrays.copyOf(tokens, length);
                float[] expected = exact.apply(prefix).rowData(0);
                assertArrayEquals(expected, rolling.apply(prefix).rowData(0));
                metal.releaseTemporaryResources();
            }
        }
    }

    @ParameterizedTest
    @MethodSource("windows")
    void gpuWrapsMatchIndependentLocalAttentionIncludingOneSlotAndRepeatedTokens(Kind kind, int capacity) {
        int[] tokens = randomTargets(capacity * 3 + 5, 31, 997);
        if (capacity == 1) Arrays.fill(tokens, 3);
        Tensor expected = reference(kind, capacity, tokens);
        Tensor.setBackend(metal);
        DecoderOnlyModel model = InferenceTestModels.create(kind, capacity, 32);
        try (var session = model.openGeneration(ContextWindowMode.ROLLING)) {
            for (int length = 1; length <= tokens.length; length++) {
                int[] window = Arrays.copyOfRange(tokens, Math.max(0, length - capacity), length);
                Tensor row = Tensor.from2D(new float[][]{expected.rowData(length - 1)});
                assertTensorClose(row, session.apply(window), 2e-5f, 2e-4f);
                metal.releaseTemporaryResources();
            }
        }
    }

    private static Tensor reference(Kind kind, int capacity, int[] tokens) {
        Tensor.setBackend(SlidingWindowReference.backend(capacity));
        DecoderOnlyModel model = InferenceTestModels.create(kind, tokens.length, 32);
        try (var ignored = GradMode.noGrad()) { return model.forward(tokens); }
    }

    private static Stream<Arguments> windows() {
        return Arrays.stream(Kind.values()).filter(kind -> kind != Kind.ORIGIN)
                .flatMap(kind -> Stream.of(1, 4, 33).map(capacity -> Arguments.of(kind, capacity)));
    }
}
