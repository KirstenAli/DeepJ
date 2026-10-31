package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.InferenceTestModels;
import io.github.kirstenali.deepj.testing.InferenceTestModels.Kind;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class CachedGenerationTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(Kind.class)
    void everyIncrementalLogitMatchesItsFullPrefixIncludingContextResets(Kind kind) {
        DecoderOnlyModel model = InferenceTestModels.create(kind, 8, 16);
        int[] tokens = {1, 7, 2, 11, 4, 19, 3, 5, 13, 17, 23};
        try (GenerationSession session = model.openGeneration(); var ignored = GradMode.noGrad()) {
            for (int length = 1; length <= tokens.length; length++) {
                int[] prefix = Arrays.copyOfRange(tokens, Math.max(0, length - 8), length);
                assertLastLogits(model.forward(prefix), session.apply(prefix));
                Tensor.backend().releaseTemporaryResources();
            }
        }
    }

    @ParameterizedTest
    @EnumSource(Kind.class)
    void independentInterleavedRequestsAndChangedPromptsDoNotReuseEachOthersCache(Kind kind) {
        DecoderOnlyModel model = InferenceTestModels.create(kind, 8, 16);
        try (var first = model.openGeneration(); var second = model.openGeneration();
             var ignored = GradMode.noGrad()) {
            first.apply(new int[]{1, 2});
            second.apply(new int[]{8, 9, 10});
            assertLastLogits(model.forward(new int[]{1, 2, 3}), first.apply(new int[]{1, 2, 3}));
            assertLastLogits(model.forward(new int[]{8, 9, 10, 11}), second.apply(new int[]{8, 9, 10, 11}));
            assertLastLogits(model.forward(new int[]{9, 2, 3, 4}), first.apply(new int[]{9, 2, 3, 4}));
        }
    }

    @Test
    void sessionsRejectChangedWeightsAndBackendsAndCannotBeUsedAfterClosing() {
        DecoderOnlyModel model = InferenceTestModels.create(Kind.NOVA, 8, 16);
        GenerationSession session = model.openGeneration();
        session.apply(new int[]{1, 2});
        try (var ignored = GradMode.noGrad()) { model.parameters().get(0).value.addScalarInPlace(0.1f); }
        assertThrows(IllegalStateException.class, () -> session.apply(new int[]{1, 2, 3}));
        session.close();
        session.close();
        assertThrows(IllegalStateException.class, () -> session.apply(new int[]{1}));
        try (var fresh = model.openGeneration()) {
            Tensor.setBackend(new CpuBackend());
            assertThrows(IllegalStateException.class, () -> fresh.apply(new int[]{1}));
        }
    }

    @Test
    void invalidContextsAndFailedForwardDoNotLeaveAReusablePartialCache() {
        DecoderOnlyModel model = InferenceTestModels.create(Kind.LOOP, 8, 16);
        try (var session = model.openGeneration()) {
            assertThrows(IllegalArgumentException.class, () -> session.apply(null));
            assertThrows(IllegalArgumentException.class, () -> session.apply(new int[0]));
            assertThrows(IllegalArgumentException.class, () -> session.apply(new int[9]));
            assertThrows(RuntimeException.class, () -> session.apply(new int[]{31}));
            assertThrows(IllegalStateException.class, () -> session.apply(new int[]{1}));
        }
    }

    @ParameterizedTest
    @EnumSource(Kind.class)
    void closingInferenceDoesNotChangeParametersOrBreakTrainingAutograd(Kind kind) {
        DecoderOnlyModel model = InferenceTestModels.create(kind, 8, 16);
        float[] weights = model.parameters().get(0).value.rawData().clone();
        try (var session = model.openGeneration()) {
            session.apply(new int[]{1, 2});
            assertFalse(session.apply(new int[]{1, 2, 3}).requiresGrad());
        }
        model.forward(new int[]{1, 2, 3}).sumTensor().backward();
        assertNotNull(model.parameters().get(0).grad());
        assertArrayEquals(weights, model.parameters().get(0).value.rawData());
    }

    private static void assertLastLogits(Tensor expected, Tensor actual) {
        assertArrayEquals(expected.rowData(expected.matrixRows() - 1), actual.rowData(actual.matrixRows() - 1), 2e-5f);
    }
}
