package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.*;

class MetalNarrowMixedMatmulTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void narrowMixedProductsPreserveTheFullFloat32RangeAndMantissa(DType dtype) {
        Tensor compact = identity(dtype, 64);
        Tensor full = values(1024, 64);
        Tensor output = metal.matmul(full, compact);
        assertEquals(DType.FLOAT32, output.dtype());
        assertArrayEquals(full.toFloatArray(), output.toFloatArray());
        Tensor left = identity(dtype, 1024);
        assertArrayEquals(full.toFloatArray(), metal.matmul(left, full).toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void sharedBatchesAndTransposesAgreeWithExplicitFloat32Products(DType dtype) {
        Tensor full = Tensor.from(random(3072, 64, 653).toFloatArray(), 3, 1024, 64);
        Tensor compact = Tensor.from(random(192, 64, 659).toFloatArray(), dtype, 3, 64, 64);
        Tensor expanded = Tensor.from(compact.toFloatArray(), compact.shape());
        assertTensorClose(metal.matmul(full, expanded), metal.matmul(full, compact), 3e-5f, 5e-4f);
        Tensor transposed = metal.permute(full, 0, 2, 1);
        Tensor right = Tensor.from(random(3072, 64, 661).toFloatArray(), dtype, 3, 1024, 64);
        Tensor reference = metal.matmul(transposed, Tensor.from(right.toFloatArray(), right.shape()));
        assertTensorClose(reference, metal.matmul(transposed, right), 3e-5f, 5e-4f);
    }

    private static Tensor identity(DType dtype, int width) {
        float[] values = new float[width * width];
        for (int index = 0; index < width; index++) { values[index * (width + 1)] = 1; }
        return Tensor.from(values, dtype, width, width);
    }

    private static Tensor values(int rows, int columns) {
        float[] values = new float[rows * columns];
        for (int index = 0; index < values.length; index++) {
            values[index] = index % 2 == 0 ? 131072 : 1 + Math.scalb(1.0f, -20);
        }
        return Tensor.from(values, rows, columns);
    }
}
