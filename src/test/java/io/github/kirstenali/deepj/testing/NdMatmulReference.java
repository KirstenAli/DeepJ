package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/** Independent coordinate enumeration and double accumulation; no production matmul/layout helpers. */
public final class NdMatmulReference {
    private NdMatmulReference() {}

    public static Tensor broadcastLeft(DType dtype) {
        return Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, dtype, 2, 1, 2, 3);
    }

    public static Tensor broadcastRight(DType dtype) {
        return Tensor.from(new float[]{1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 2,
                3, 0, 0, 3, 3, 3}, dtype, 1, 3, 3, 2);
    }

    public static void assertBroadcastFixture(DType dtype) {
        Tensor output = broadcastLeft(dtype).matmul(broadcastRight(dtype));
        assertArrayEquals(new int[]{2, 3, 2, 2}, output.shape());
        assertArrayEquals(new float[]{4, 5, 10, 11, 8, 10, 20, 22, 12, 15, 30, 33,
                16, 17, 22, 23, 32, 34, 44, 46, 48, 51, 66, 69}, output.toFloatArray());
    }

    public static void assertProduct(Tensor left, Tensor right, int... expectedShape) {
        Operand a = new Operand(left, true), b = new Operand(right, false);
        int matrixAxes = (left.rank() == 1 ? 0 : 1) + (right.rank() == 1 ? 0 : 1);
        int[] batch = Arrays.copyOf(expectedShape, expectedShape.length - matrixAxes);
        int count = Arrays.stream(expectedShape).reduce(1, Math::multiplyExact);
        float[] expected = reference(a, b, batch, count);
        DType dtype = left.dtype() == right.dtype() ? left.dtype() : DType.FLOAT32;
        expected = Tensor.from(expected, dtype, expectedShape).toFloatArray();
        Tensor actual = left.matmul(right);
        assertArrayEquals(expectedShape, actual.shape());
        assertEquals(dtype, actual.dtype());
        assertArrayEquals(expected, actual.toFloatArray(), 3e-6f);
        assertEquals((long) count * dtype.bytes(), actual.storageBytes());
    }

    private static float[] reference(Operand a, Operand b, int[] batchShape, int count) {
        float[] expected = new float[count];
        for (int index = 0; index < count; index++) {
            int[] batch = coordinate(index / (a.rows * b.columns), batchShape);
            int row = index / b.columns % a.rows, column = index % b.columns;
            double sum = 0;
            for (int inner = 0; inner < a.columns; inner++) {
                sum += (double) a.at(batch, row, inner) * b.at(batch, inner, column);
            }
            expected[index] = (float) sum;
        }
        return expected;
    }

    private static int[] coordinate(int flat, int[] dimensions) {
        int[] result = new int[dimensions.length];
        for (int axis = result.length - 1; axis >= 0; axis--) {
            result[axis] = flat % dimensions[axis];
            flat /= dimensions[axis];
        }
        return result;
    }

    private static final class Operand {
        private final int[] shape;
        private final float[] values;
        private final boolean left;
        private final int rows, columns;

        Operand(Tensor tensor, boolean left) {
            this.shape = tensor.shape();
            this.values = tensor.toFloatArray();
            this.left = left;
            rows = tensor.rank() == 1 ? (left ? 1 : shape[0]) : shape[shape.length - 2];
            columns = tensor.rank() == 1 ? (left ? shape[0] : 1) : shape[shape.length - 1];
        }

        float at(int[] batch, int row, int column) {
            if (shape.length == 1) return values[left ? column : row];
            int index = 0, leading = batch.length - shape.length + 2;
            for (int axis = 0; axis < shape.length - 2; axis++) {
                index = index * shape[axis] + (shape[axis] == 1 ? 0 : batch[leading + axis]);
            }
            return values[(index * rows + row) * columns + column];
        }
    }
}
