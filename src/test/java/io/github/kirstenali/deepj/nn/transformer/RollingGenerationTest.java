package io.github.kirstenali.deepj.nn.transformer;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.InferenceTestModels;
import io.github.kirstenali.deepj.testing.InferenceTestModels.Kind;
import io.github.kirstenali.deepj.testing.SlidingWindowReference;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.stream.Stream;
import org.junit.jupiter.params.provider.Arguments;

import static org.junit.jupiter.api.Assertions.*;

class RollingGenerationTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(value = Kind.class, names = "ORIGIN", mode = EnumSource.Mode.EXCLUDE)
    void modesHaveExactlyTheSameLogitsBeforeTheFirstEviction(Kind kind) {
        DecoderOnlyModel model = InferenceTestModels.create(kind, 8, 16);
        int[] tokens = {1, 7, 2, 11, 4, 19, 3, 5};
        try (var exact = model.openGeneration(); var rolling = model.openGeneration(ContextWindowMode.ROLLING)) {
            for (int length = 3; length <= tokens.length; length++) {
                int[] prefix = Arrays.copyOf(tokens, length);
                assertArrayEquals(exact.apply(prefix).rowData(0), rolling.apply(prefix).rowData(0));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("windows")
    void wrappedCachesMatchIndependentFullSequenceSlidingAttention(Kind kind, int capacity, boolean repeated) {
        int[] tokens = new int[capacity * 3 + 5];
        for (int index = 0; index < tokens.length; index++) {
            tokens[index] = repeated ? 3 : (index * 7 + 1) % 31;
        }
        float[][] expected = reference(kind, capacity, tokens);
        Tensor.setBackend(new CpuBackend());
        DecoderOnlyModel model = InferenceTestModels.create(kind, capacity, 16);
        try (var session = model.openGeneration(ContextWindowMode.ROLLING)) {
            for (int length = 1; length <= tokens.length; length++) {
                int[] window = Arrays.copyOfRange(tokens, Math.max(0, length - capacity), length);
                assertArrayEquals(expected[length - 1], session.apply(window).rowData(0), 2e-5f);
            }
        }
    }

    private static float[][] reference(Kind kind, int capacity, int[] tokens) {
        Tensor.setBackend(SlidingWindowReference.backend(capacity));
        DecoderOnlyModel model = InferenceTestModels.create(kind, tokens.length, 16);
        float[][] result = new float[tokens.length][];
        try (var ignored = GradMode.noGrad()) {
            Tensor logits = model.forward(tokens);
            for (int row = 0; row < result.length; row++) {
                result[row] = logits.rowData(row);
            }
        }
        return result;
    }

    private static Stream<Arguments> windows() {
        return Arrays.stream(Kind.values()).filter(kind -> kind != Kind.ORIGIN)
                .flatMap(kind -> Stream.of(1, 4, 8).flatMap(capacity -> Stream.of(false, true)
                        .map(repeated -> Arguments.of(kind, capacity, repeated))));
    }

    @Test
    void changedPromptRestartsPositionsAndIndependentSessionsDoNotShareHistory() {
        DecoderOnlyModel model = InferenceTestModels.create(Kind.LOOP, 4, 16);
        try (var first = model.openGeneration(ContextWindowMode.ROLLING);
             var fresh = model.openGeneration(ContextWindowMode.ROLLING)) {
            first.apply(new int[]{1, 2, 3, 4});
            first.apply(new int[]{2, 3, 4, 5});
            first.apply(new int[]{3, 4, 5, 6});
            int[] changed = {8, 9, 10, 11};
            assertArrayEquals(fresh.apply(changed).rowData(0), first.apply(changed).rowData(0));
        }
    }

    @Test
    void originRejectsRollingAndNullModesRatherThanSilentlyChangingPositions() {
        DecoderOnlyModel origin = InferenceTestModels.create(Kind.ORIGIN, 4, 16);
        assertThrows(UnsupportedOperationException.class, () -> origin.openGeneration(ContextWindowMode.ROLLING));
        assertThrows(NullPointerException.class, () -> origin.openGeneration(null));
    }

    @Test
    void rolledSessionRejectsWeightChangesAndInvalidContextsAndClosesAfterForwardFailure() {
        DecoderOnlyModel model = InferenceTestModels.create(Kind.PRISM, 2, 16);
        try (var session = model.openGeneration(ContextWindowMode.ROLLING)) {
            session.apply(new int[]{1, 2});
            session.apply(new int[]{2, 3});
            assertThrows(IllegalArgumentException.class, () -> session.apply(new int[3]));
            assertThrows(RuntimeException.class, () -> session.apply(new int[]{3, 31}));
            assertThrows(IllegalStateException.class, () -> session.apply(new int[]{3, 4}));
        }
        try (var session = model.openGeneration(ContextWindowMode.ROLLING); var ignored = GradMode.noGrad()) {
            model.parameters().get(0).value.addScalarInPlace(0.1f);
            assertThrows(IllegalStateException.class, () -> session.apply(new int[]{1}));
        }
    }
}
