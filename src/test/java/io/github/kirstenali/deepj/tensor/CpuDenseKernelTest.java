package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class CpuDenseKernelTest {
    private final CpuBackend cpu = new CpuBackend();
    @BeforeEach void backend() { Tensor.setBackend(cpu); }

    @Test
    void densePointwiseKernelsPreserveRanksOffsetsAndEveryFloatBit() {
        for (int[] shape : new int[][]{{}, {8193}, {3, 1, 2731}, {1, 3, 1, 1, 2731, 1}, {2, 0, 3}}) {
            Tensor left = input(shape, 1), right = input(shape, 2);
            float[] a = left.toFloatArray(), b = right.toFloatArray();
            Tensor sum = left.add(right), scaled = left.multiplyScalar(-0.125f);
            assertArrayEquals(shape, sum.shape());
            assertBits(expected(a, b, Float::sum), sum.toFloatArray());
            assertBits(expected(a, b, (x, y) -> x * -0.125f), scaled.toFloatArray());
            left.addInPlace(right);
            assertBits(sum.toFloatArray(), left.toFloatArray());
        }
    }

    @Test
    void overlappingDenseOffsetViewsReadTheOriginalValuesBeforeParallelWrites() {
        Tensor parent = Tensor.from(sequence(8196));
        Tensor source = parent.slice(0, 0, 8193), target = parent.slice(0, 1, 8194);
        float[] a = target.toFloatArray(), b = source.toFloatArray();
        target.addInPlace(source);
        assertBits(expected(a, b, Float::sum), target.toFloatArray());
        assertEquals(0, parent.get(0));
        assertEquals(8194, parent.get(8194));
        assertEquals(8195, parent.get(8195));
    }

    @Test
    void denseSwiGluForwardAndAdjointsMatchIndependentScalarArithmetic() {
        for (int[] shape : new int[][]{{}, {8193}, {3, 1, 2731}, {1, 3, 1, 2731}, {0, 3}}) {
            Tensor gate = input(shape, 3), up = input(shape, 4), gradient = input(shape, 5);
            assertSwiGlu(gate, up, gradient);
        }
        Tensor gate = Tensor.from(new float[]{-100, -3, -0.0f, 0, 1, 20, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.NaN}, 1, 3, 3);
        assertSwiGlu(gate, Tensor.ones(1, 9).reshape(1, 3, 3), Tensor.ones(1, 9).reshape(1, 3, 3));
    }

    @Test
    void serialAndParallelDenseResultsHaveIdenticalBits() {
        boolean previous = DeepJExecutor.isParallelEnabled();
        try {
            Tensor gate = input(new int[]{3, 1, 2731}, 6), up = input(gate.shape(), 7);
            DeepJExecutor.setParallelEnabled(false);
            float[] expected = cpu.swiGlu(gate, up).toFloatArray();
            DeepJExecutor.setParallelEnabled(true);
            assertBits(expected, cpu.swiGlu(gate, up).toFloatArray());
            assertSwiGlu(gate, up, input(gate.shape(), 8));
        } finally {
            DeepJExecutor.setParallelEnabled(previous);
        }
    }

    @Test
    void noncontiguousBroadcastAndCompactInputsKeepTheirGeneralPaths() {
        Tensor gate = input(new int[]{3, 1, 2731}, 9).permute(2, 1, 0);
        Tensor up = input(new int[]{3, 1, 2731}, 10).permute(2, 1, 0);
        assertSwiGlu(gate, up, input(gate.shape(), 11));
        assertBits(expected(gate.toFloatArray(), up.toFloatArray(), Float::sum), gate.add(up).toFloatArray());
        assertArrayEquals(new float[]{3, 4, 5, 6}, Tensor.from(new float[]{1, 2, 3, 4}, 2, 1, 2)
                .add(Tensor.from(new float[]{2})).toFloatArray());
        for (DType type : new DType[]{DType.FLOAT16, DType.BFLOAT16}) {
            Tensor compact = Tensor.from(new float[]{1, 2, 3, 4}, type, 2, 1, 2);
            assertEquals(type, cpu.swiGlu(compact, compact).dtype());
            assertEquals(DType.FLOAT32, cpu.swiGluBackward(compact, compact, compact).gateGradient().dtype());
        }
    }

    @Test
    void matrixScalarKernelsMatchEveryBitForBothExecutionModes() {
        boolean previous = DeepJExecutor.isParallelEnabled();
        try {
            for (boolean parallel : new boolean[]{false, true}) {
                DeepJExecutor.setParallelEnabled(parallel);
                assertMatrixScalarBits();
            }
        } finally {
            DeepJExecutor.setParallelEnabled(previous);
        }
    }

    private void assertMatrixScalarBits() {
        Tensor matrix = input(new int[]{576, 1664}, 12).contiguous();
        float[] values = matrix.toFloatArray();
        for (float scalar : new float[]{-0.125f, -0.0f, Float.POSITIVE_INFINITY, Float.NaN}) {
            float[] expected = expected(values, values, (left, ignored) -> left * scalar);
            assertBits(expected, cpu.multiplyScalar(matrix, scalar).toFloatArray());
            Tensor target = new Tensor(matrix);
            cpu.multiplyScalarInPlace(target, scalar);
            assertBits(expected, target.toFloatArray());
        }
    }

    private void assertSwiGlu(Tensor gate, Tensor up, Tensor gradient) {
        float[][] expected = reference(gate.toFloatArray(), up.toFloatArray(), gradient.toFloatArray());
        var result = cpu.swiGluBackward(gradient, gate, up);
        assertArrayEquals(gate.shape(), result.gateGradient().shape());
        assertBits(expected[0], cpu.swiGlu(gate, up).toFloatArray());
        assertBits(expected[1], result.gateGradient().toFloatArray());
        assertBits(expected[2], result.upGradient().toFloatArray());
    }

    private static float[][] reference(float[] a, float[] b, float[] dy) {
        float[] forward = new float[a.length], da = new float[a.length], db = new float[a.length];
        for (int index = 0; index < a.length; index++) {
            float s = 1.0f / (1.0f + (float) Math.exp(-a[index])), activated = a[index] * s;
            forward[index] = activated * b[index];
            da[index] = (dy[index] * b[index]) * (s + activated * (1.0f - s));
            db[index] = dy[index] * activated;
        }
        return new float[][]{forward, da, db};
    }

    private static Tensor input(int[] shape, int seed) {
        int count = new TensorShape(shape).elements();
        float[] values = new float[count + 2];
        var random = new Random(seed);
        for (int index = 0; index < values.length; index++) {
            values[index] = random.nextFloat() * 8 - 4;
        }
        return Tensor.from(values).slice(0, 1, count + 1).reshape(shape);
    }

    private static float[] sequence(int count) {
        float[] values = new float[count];
        for (int index = 0; index < count; index++) {
            values[index] = index;
        }
        return values;
    }

    private static float[] expected(float[] left, float[] right, CpuNdElementwise.Binary operation) {
        float[] result = new float[left.length];
        for (int index = 0; index < result.length; index++) {
            result[index] = operation.apply(left[index], right[index]);
        }
        return result;
    }

    private static void assertBits(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int index = 0; index < expected.length; index++) {
            if (Float.isNaN(expected[index])) assertTrue(Float.isNaN(actual[index]));
            else assertEquals(Float.floatToRawIntBits(expected[index]), Float.floatToRawIntBits(actual[index]));
        }
    }
}
