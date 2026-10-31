package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Random;

import static io.github.kirstenali.deepj.testing.NdMatmulReference.assertProduct;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalNative16MatmulTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void nativeMatrixProductsHandleInteriorAndAllTileEdges(DType dtype) {
        assumeTrue(MetalBackend.supportsNative16Matmul(dtype), "Native16 matrix capability unavailable");
        assertProduct(values(dtype, 64, 48), values(dtype, 48, 64), 64, 64);
        assertProduct(values(dtype, 33, 37), values(dtype, 37, 35), 33, 35);
        assertProduct(values(dtype, 1, 17), values(dtype, 17, 1), 1, 1);
    }

    @Test
    void mixedInputsPreserveFloat32RangeAndDoNotNarrowToHalf() {
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            Tensor large = Tensor.from(new float[]{131072, -131072, 1.0001f, 2.0001f}, 2, 2);
            Tensor compact = Tensor.from(new float[]{1, 0, 0, 1}, dtype, 2, 2);
            assertArrayEquals(large.toFloatArray(), large.matmul(compact).toFloatArray());
            assertArrayEquals(large.toFloatArray(), compact.matmul(large).toFloatArray());
        }
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactProductsAccumulateBeforeRoundingTheOutput(DType dtype) {
        Tensor left = Tensor.from(new float[]{4096, 1, -4096}, dtype, 1, 3);
        Tensor right = Tensor.from(new float[]{1, 1, 1}, dtype, 3, 1);
        assertArrayEquals(new float[]{1}, left.matmul(right).toFloatArray());
        assertProduct(values(dtype, 7, 1025), values(dtype, 1025, 9), 7, 9);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void batchesBroadcastsOffsetsAndTransposedMatricesHaveIndependentProducts(DType dtype) {
        Tensor left = values(dtype, 2, 35, 39).slice(1, 1, 34, 1).slice(2, 1, 38, 1);
        Tensor right = values(dtype, 35, 37).transpose(0, 1);
        assertProduct(left, right, 2, 33, 35);
        assertProduct(left.transpose(1, 2), left, 2, 37, 37);
        assertProduct(left, right.to(DType.FLOAT32), 2, 33, 35);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void mixedBackwardProductsRetainFloat32Gradients(DType dtype) {
        Tensor left = values(dtype, 33, 37).requiresGrad(true);
        Tensor right = values(dtype, 37, 35).requiresGrad(true);
        Tensor expectedLeft = cpu.matmul(Tensor.ones(33, 35), cpu.cast(right, DType.FLOAT32).transpose(0, 1));
        Tensor expectedRight = cpu.matmul(cpu.cast(left, DType.FLOAT32).transpose(0, 1), Tensor.ones(33, 35));
        left.matmul(right).sumTensor().backward();
        assertEquals(DType.FLOAT32, left.grad().dtype());
        assertEquals(DType.FLOAT32, right.grad().dtype());
        assertArrayEquals(expectedLeft.toFloatArray(), left.grad().toFloatArray());
        assertArrayEquals(expectedRight.toFloatArray(), right.grad().toFloatArray());
    }

    private static Tensor values(DType dtype, int... shape) {
        int count = 1;
        for (int dimension : shape) { count = Math.multiplyExact(count, dimension); }
        float[] values = new float[count];
        Random random = new Random(count + 20261026L);
        for (int index = 0; index < count; index++) { values[index] = random.nextInt(-7, 8) / 8.0f; }
        return Tensor.from(values, dtype, shape);
    }
}
