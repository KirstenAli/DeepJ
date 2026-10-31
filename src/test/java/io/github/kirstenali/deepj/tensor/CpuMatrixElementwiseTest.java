package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import static org.junit.jupiter.api.Assertions.*;

class CpuMatrixElementwiseTest {
    private final CpuBackend cpu = new CpuBackend();
    private static final int[][] SHAPES = {{97, 193}, {193, 97}, {1, 23}, {23, 1}, {0, 3}, {3, 0}};

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void unaryMatrixKernelsKeepIndependentScalarBitsAndBackingStorage(boolean parallel) {
        withParallel(parallel, () -> {
            for (int[] shape : SHAPES) {
                assertUnary(shape, cpu::negInPlace, value -> -value);
                assertUnary(shape, cpu::sqrtInPlace, value -> (float) Math.sqrt(value));
            }
        });
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void binaryMatrixKernelsKeepIndependentScalarBitsAndBackingStorage(boolean parallel) {
        withParallel(parallel, () -> {
            for (int[] shape : SHAPES) {
                assertBinary(shape, cpu::addInPlace, Float::sum);
                assertBinary(shape, cpu::subtractInPlace, (left, right) -> left - right);
                assertBinary(shape, cpu::multiplyInPlace, (left, right) -> left * right);
                assertBinary(shape, cpu::divideInPlace, (left, right) -> left / right);
            }
        });
    }

    private void assertUnary(int[] shape, Consumer<Tensor> operation, CpuNdElementwise.Unary reference) {
        Tensor tensor = values(shape, 0);
        float[] expected = tensor.toFloatArray(), backing = tensor.rawData();
        for (int index = 0; index < expected.length; index++) {
            expected[index] = reference.apply(expected[index]);
        }
        operation.accept(tensor);
        assertSame(backing, tensor.rawData());
        assertArrayEquals(shape, tensor.shape());
        assertBits(expected, tensor.toFloatArray());
    }

    private void assertBinary(int[] shape, BiConsumer<Tensor, Tensor> operation, CpuNdElementwise.Binary reference) {
        Tensor left = values(shape, 0), right = values(shape, 3);
        float[] expected = left.toFloatArray(), other = right.toFloatArray(), backing = left.rawData();
        for (int index = 0; index < expected.length; index++) {
            expected[index] = reference.apply(expected[index], other[index]);
        }
        operation.accept(left, right);
        assertSame(backing, left.rawData());
        assertArrayEquals(shape, left.shape());
        assertBits(expected, left.toFloatArray());
        assertBits(other, right.toFloatArray());
    }

    @Test
    void aliasingAndLogicalViewsRetainTheirExistingSemantics() {
        Tensor.setBackend(cpu);
        Tensor parent = Tensor.from(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2);
        Tensor middle = parent.slice(0, 1, 3);
        cpu.addInPlace(middle, middle);
        cpu.negInPlace(middle.permute(1, 0));
        assertArrayEquals(new float[]{1, 2, -6, -8, -10, -12, 7, 8}, parent.toFloatArray());
        Tensor identical = values(new int[]{7, 13}, 0);
        float[] before = identical.toFloatArray();
        cpu.multiplyInPlace(identical, identical);
        for (int index = 0; index < before.length; index++) {
            before[index] *= before[index];
        }
        assertBits(before, identical.toFloatArray());
    }

    private static Tensor values(int[] shape, int shift) {
        float[] palette = {-0.0f, 0, Float.MIN_VALUE, -Float.MIN_VALUE, 0.25f, -3.125f,
                Float.MAX_VALUE, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN};
        float[] values = new float[shape[0] * shape[1]];
        for (int index = 0; index < values.length; index++) {
            values[index] = palette[(index + shift) % palette.length];
        }
        return Tensor.from(values, shape);
    }

    private void withParallel(boolean parallel, Runnable test) {
        boolean previous = DeepJExecutor.isParallelEnabled();
        Tensor.setBackend(cpu);
        try {
            DeepJExecutor.setParallelEnabled(parallel);
            test.run();
        } finally {
            DeepJExecutor.setParallelEnabled(previous);
        }
    }

    private static void assertBits(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int index = 0; index < expected.length; index++) {
            assertEquals(Float.floatToIntBits(expected[index]), Float.floatToIntBits(actual[index]));
        }
    }
}
