package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalNative16PrecisionTest extends MetalDifferentialTestSupport {
    @Test
    void halfSubnormalInputsAndProductsSurviveFloat32Accumulation() {
        assumeTrue(MetalBackend.supportsNative16Matmul(DType.FLOAT16));
        float minimum = Math.scalb(1.0f, -24);
        Tensor left = Tensor.from(new float[]{minimum, minimum}, DType.FLOAT16, 1, 2);
        Tensor right = Tensor.from(new float[]{0.5f, 0.5f}, DType.FLOAT16, 2, 1);
        assertArrayEquals(new float[]{minimum}, left.matmul(right).toFloatArray());
    }

    @Test
    void bfloatProductsRetainTheirWiderExponentRange() {
        assumeTrue(MetalBackend.supportsNative16Matmul(DType.BFLOAT16));
        Tensor left = Tensor.from(new float[]{Math.scalb(1.0f, -120), Math.scalb(1.0f, 120)},
                DType.BFLOAT16, 1, 2);
        Tensor right = Tensor.from(new float[]{Math.scalb(1.0f, -6), Math.scalb(1.0f, -120)},
                DType.BFLOAT16, 2, 1);
        assertArrayEquals(new float[]{1}, left.matmul(right).toFloatArray());
        assertArrayEquals(new float[]{Math.scalb(1.0f, -126)},
                left.slice(1, 0, 1, 1).matmul(right.slice(0, 0, 1, 1)).toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void partialTilesPreserveValidInfinityAndNanResults(DType dtype) {
        Tensor infinity = Tensor.from(new float[]{Float.POSITIVE_INFINITY}, dtype, 1, 1);
        Tensor one = Tensor.from(new float[]{1}, dtype, 1, 1);
        Tensor zero = Tensor.from(new float[]{0}, dtype, 1, 1);
        assertArrayEquals(new float[]{Float.POSITIVE_INFINITY}, infinity.matmul(one).toFloatArray());
        assertTrue(Float.isNaN(infinity.matmul(zero).get(0, 0)));
        Tensor nan = Tensor.from(new float[]{Float.NaN}, dtype, 1, 1);
        assertTrue(Float.isNaN(nan.matmul(one).get(0, 0)));
    }
}
