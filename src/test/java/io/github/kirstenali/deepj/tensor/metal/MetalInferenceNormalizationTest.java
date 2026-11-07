package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.autograd.NormalizationOps;
import io.github.kirstenali.deepj.tensor.*;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalInferenceNormalizationTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void outputOnlyInferenceMatchesSavedStatisticsForStridedInputsAndMixedGains(DType dtype) {
        for (int width : new int[]{3, 127, 128, 129, 2048}) {
            Tensor input = Tensor.random(dtype, new Random(12), 3, width * 2)
                    .slice(1, 0, width * 2, 2).requiresGrad(true);
            for (DType gainType : new DType[]{dtype, DType.FLOAT32}) { compare(input, gainType, width); }
        }
    }

    private void compare(Tensor input, DType gainType, int width) {
        Tensor gain = Tensor.random(gainType, new Random(17), width).requiresGrad(true);
        Tensor bias = Tensor.random(gainType, new Random(19), width).requiresGrad(true);
        for (boolean centered : new boolean[]{false, true}) {
            Tensor expected = metal.normalize(input, gain, centered ? bias : null, 1e-5f, centered).output();
            try (var ignored = GradMode.noGrad()) {
                Tensor actual = centered ? NormalizationOps.layerNorm(input, gain, bias, 1e-5f)
                        : NormalizationOps.rmsNorm(input, gain, 1e-5f);
                assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
                assertEquals(expected.dtype(), actual.dtype());
                assertFalse(actual.requiresGrad());
            }
        }
        assertNull(input.grad());
        assertNull(gain.grad());
    }

    @Test void inferenceRequiresNoGradAndPreservesEmptyShapes() {
        Tensor input = Tensor.ones(DType.FLOAT16, 2, 128), gain = Tensor.ones(DType.FLOAT16, 128);
        assertThrows(IllegalStateException.class, () -> metal.normalizeForInference(input, gain, null, 1e-5f, false));
        try (var ignored = GradMode.noGrad()) {
            Tensor empty = NormalizationOps.rmsNorm(Tensor.zeros(DType.FLOAT16, 0, 128), gain, 1e-5f);
            assertArrayEquals(new int[]{0, 128}, empty.shape());
            assertEquals(0, empty.toFloatArray().length);
            assertThrows(IllegalArgumentException.class,
                    () -> metal.normalizeForInference(input, gain, null, Float.NaN, false));
        }
    }
}
