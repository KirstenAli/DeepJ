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
        assertScalars(new Tensor(DType.FLOAT16, 3));
        assertScalars(new Tensor(DType.BFLOAT16, 3));
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

    @Test
    void elementCountBinaryKernelsPreserveRankZeroThroughSixWithoutHostOutputs() {
        for (int rank = 0; rank <= 6; rank++) {
            int[] shape = new int[rank];
            Arrays.fill(shape, 2);
            float[] values = new float[1 << rank], divisors = new float[values.length];
            for (int index = 0; index < values.length; index++) { values[index] = index - 3.0f; }
            Arrays.fill(divisors, 2);
            assertBinary(Tensor.from(values, shape), Tensor.from(divisors, shape));
        }
    }

    @Test
    void binaryViewsBroadcastsEmptyAxesAndCompactStorageRetainTheirGeneralPaths() {
        Tensor source = Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertBinary(source.slice(0, 1, 2, 1), source.slice(0, 0, 1, 1));
        assertBinary(source.permute(1, 0), source.permute(1, 0));
        assertBinary(source, Tensor.from(new float[]{2, 4, 8}, 1, 3));
        assertBinary(new Tensor(2, 0, 3), new Tensor(2, 0, 3));
        for (DType dtype : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            assertBinary(Tensor.from(new float[]{1, 2}, dtype, 1, 2), source.slice(1, 0, 2, 1));
        }
    }

    private void assertBinary(Tensor left, Tensor right) {
        Tensor[] expected = {cpu.add(left, right), cpu.subtract(left, right),
                cpu.multiply(left, right), cpu.divide(left, right)};
        Tensor[] actual = {metal.add(left, right), metal.subtract(left, right),
                metal.multiply(left, right), metal.divide(left, right)};
        for (int index = 0; index < actual.length; index++) {
            assertFalse(TensorTestStorage.hasHostStorage(actual[index]));
            assertArrayEquals(expected[index].shape(), actual[index].shape());
            assertArrayEquals(expected[index].toFloatArray(), actual[index].toFloatArray());
        }
    }
}
