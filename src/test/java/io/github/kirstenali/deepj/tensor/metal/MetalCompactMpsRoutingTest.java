package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;


import static org.junit.jupiter.api.Assertions.*;

/** Large BF16 products stage through FLOAT32 for MPS; FP16 reaches MPS on either side. */
class MetalCompactMpsRoutingTest extends MetalDifferentialTestSupport {
    @Test
    void bfloatProductsKeepTheirResultTypeAndMatchTheFloat32Reference() {
        Tensor left = compact(DType.BFLOAT16, 256, 1024, 11);
        Tensor right = compact(DType.BFLOAT16, 1024, 512, 13);
        assertTrue(MetalMixedMatmul.stageBfloat(left, right));
        Tensor output = metal.matmul(left, right);
        assertEquals(DType.BFLOAT16, output.dtype());
        Tensor reference = metal.cast(metal.matmul(widen(left), widen(right)), DType.BFLOAT16);
        assertTensorClose(reference, output, 1e-6f, 1e-5f);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactLeftOperandsAgreeWithTheFloat32Product(DType dtype) {
        Tensor left = compact(dtype, 256, 128, 17);
        Tensor right = random(128, 320, 19);
        Tensor output = metal.matmul(left, right);
        assertEquals(DType.FLOAT32, output.dtype());
        assertTensorClose(metal.matmul(widen(left), right), output, 1e-5f, 1e-4f);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void transposedCompactLeftOperandsMatchTheFloat32Reference(DType dtype) {
        Tensor activations = compact(dtype, 1024, 576, 23);
        Tensor upstream = random(1024, 64, 29);
        Tensor transposed = metal.permute(activations, 1, 0);
        Tensor output = metal.matmul(transposed, upstream);
        assertTensorClose(metal.matmul(widen(transposed), upstream), output, 1e-5f, 1e-4f);
    }

    @Test
    void smallBfloatProductsStayOnTheCompactKernel() {
        Tensor left = compact(DType.BFLOAT16, 8, 576, 31);
        Tensor right = compact(DType.BFLOAT16, 576, 64, 37);
        assertFalse(MetalMixedMatmul.stageBfloat(left, right));
        assertEquals(DType.BFLOAT16, metal.matmul(left, right).dtype());
    }

    private Tensor compact(DType dtype, int rows, int columns, long seed) {
        return Tensor.from(random(rows, columns, seed).toFloatArray(), dtype, rows, columns);
    }

    private Tensor widen(Tensor operand) { return metal.cast(operand, DType.FLOAT32); }
}
