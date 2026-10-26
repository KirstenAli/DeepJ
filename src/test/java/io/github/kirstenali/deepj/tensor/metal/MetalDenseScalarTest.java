package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorTestStorage;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class MetalDenseScalarTest extends MetalDifferentialTestSupport {
    @Test
    void elementCountScalarKernelsPreserveRankZeroThroughSixShapes() {
        for (int rank = 0; rank <= 6; rank++) {
            int[] shape = new int[rank];
            Arrays.fill(shape, 2);
            float[] values = new float[1 << rank];
            for (int index = 0; index < values.length; index++) { values[index] = index - 3.0f; }
            assertScalars(Tensor.from(values, shape));
        }
    }

    @Test
    void offsetScalarInputsAreCopiedOnGpuAndEmptyAxesNeedNoAllocation() {
        Tensor source = Tensor.from(new float[]{0, 1, 2, 3, 4, 5});
        assertScalars(source.slice(0, 1, 5, 1));
        assertScalars(Tensor.from(new float[]{0, 1, 2, 3, 4, 5}, 2, 3).slice(1, 0, 3, 2));
        Tensor empty = new Tensor(2, 0, 3);
        assertScalars(empty);
        assertScalars(new Tensor(0, 3));
        assertScalars(new Tensor(2, 0));
        assertThrows(UnsupportedOperationException.class, () -> metal.addScalar(new Tensor(DType.FLOAT16, 3), 1));
    }

    private void assertScalars(Tensor input) {
        Tensor sum = metal.addScalar(input, 3), product = metal.multiplyScalar(input, 2);
        Tensor quotient = metal.divideScalar(input, 2);
        assertFalse(TensorTestStorage.hasHostStorage(sum));
        assertArrayEquals(input.shape(), sum.shape());
        assertArrayEquals(input.shape(), product.shape());
        assertArrayEquals(input.shape(), quotient.shape());
        float[] original = input.toFloatArray();
        float[] sums = sum.toFloatArray(), products = product.toFloatArray(), quotients = quotient.toFloatArray();
        for (int index = 0; index < original.length; index++) {
            assertEquals(original[index] + 3, sums[index]);
            assertEquals(original[index] * 2, products[index]);
            assertEquals(original[index] / 2, quotients[index]);
        }
    }
}
