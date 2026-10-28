package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CpuMatrixScalarTest {
    private final CpuBackend cpu = new CpuBackend();
    @BeforeEach void backend() { Tensor.setBackend(cpu); }

    @Test
    void matrixScalingPreservesScalarBitsStorageAndSerialParallelResults() {
        boolean previous = DeepJExecutor.isParallelEnabled();
        try {
            for (boolean parallel : new boolean[]{false, true}) {
                DeepJExecutor.setParallelEnabled(parallel);
                assertShapes();
            }
        } finally {
            DeepJExecutor.setParallelEnabled(previous);
        }
    }

    private void assertShapes() {
        for (int[] shape : new int[][]{{768, 2112}, {2112, 768}, {1, 17}, {17, 1}, {0, 7}, {3, 0}}) {
            for (float scalar : new float[]{-0.125f, -0.0f, 0, 1, Float.POSITIVE_INFINITY, Float.NaN}) {
                assertScale(shape, scalar);
            }
        }
    }

    private void assertScale(int[] shape, float scalar) {
        float[] values = values(new TensorShape(shape).elements());
        Tensor target = Tensor.from(values, shape);
        float[] backing = target.rawData();
        cpu.multiplyScalarInPlace(target, scalar);
        assertSame(backing, target.rawData());
        assertArrayEquals(shape, target.shape());
        assertEquals(DType.FLOAT32, target.dtype());
        assertProducts(values, scalar, target.toFloatArray());
    }

    @Test
    void partialStorageAndNoncontiguousMatricesKeepTheirLogicalFallback() {
        Tensor parent = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2);
        Tensor middle = parent.slice(0, 1, 3);
        cpu.multiplyScalarInPlace(middle, 2);
        assertArrayEquals(new float[]{1, 2, 6, 8, 10, 12, 7, 8}, parent.toFloatArray());
        Tensor transposed = parent.permute(1, 0);
        cpu.multiplyScalarInPlace(transposed, -0.5f);
        assertArrayEquals(new float[]{-0.5f, -1, -3, -4, -5, -6, -3.5f, -4}, parent.toFloatArray());
    }

    @Test
    void scalarVectorNdAndCompactStorageKeepTheirExistingDtypesAndResults() {
        for (DType dtype : DType.values()) {
            for (int[] shape : new int[][]{{}, {1}, {1, 1}, {1, 1, 1}}) {
                Tensor value = Tensor.from(new float[]{2}, dtype, shape);
                cpu.multiplyScalarInPlace(value, 1.5f);
                assertEquals(dtype, value.dtype());
                assertArrayEquals(shape, value.shape());
                assertArrayEquals(new float[]{3}, value.toFloatArray());
            }
        }
    }

    private static float[] values(int count) {
        float[] special = {-0.0f, 0, Float.MIN_VALUE, -Float.MIN_VALUE, Float.MIN_NORMAL,
                Float.MAX_VALUE, -Float.MAX_VALUE, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
                Float.intBitsToFloat(0x7fc01234), 1.75f, -2.5f};
        float[] result = new float[count];
        for (int index = 0; index < count; index++) {
            result[index] = special[index % special.length];
        }
        return result;
    }

    private static void assertProducts(float[] values, float scalar, float[] actual) {
        assertEquals(values.length, actual.length);
        for (int index = 0; index < values.length; index++) {
            float expected = values[index] * scalar;
            if (Float.isNaN(expected)) assertTrue(Float.isNaN(actual[index]));
            else assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual[index]));
        }
    }
}
