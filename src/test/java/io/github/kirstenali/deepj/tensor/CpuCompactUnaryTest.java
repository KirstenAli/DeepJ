package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CpuCompactUnaryTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void contiguousOffsetsAndParallelTailsPreserveScalarResults(DType type) {
        for (int count : new int[]{0, 1, 8192, 8193, 65539}) {
            Tensor input = values(type, count + 9).slice(0, 3, count + 3).reshape(1, count);
            Tensor parent = Tensor.from(new float[count + 13], type, count + 13);
            Tensor output = parent.slice(0, 5, count + 5).reshape(1, count);
            CpuNdElementwise.unaryInto(input, output, CpuCompactUnaryTest::equation);
            assertArrayEquals(expected(input, type), output.toFloatArray());
            assertEquals(0, parent.get(4));
            assertEquals(0, parent.get(count + 5));
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void stridedAndOverlappingViewsKeepTheirOriginalValues(DType type) {
        Tensor input = values(type, 256 * 64).reshape(256, 64).transpose();
        Tensor output = new Tensor(type, 256, 64).transpose();
        CpuNdElementwise.unaryInto(input, output, CpuCompactUnaryTest::equation);
        assertArrayEquals(expected(input, type), output.toFloatArray());
        CpuNdElementwise.unaryInto(input.contiguous(), output, CpuCompactUnaryTest::equation);
        assertArrayEquals(expected(input, type), output.toFloatArray());
        Tensor parent = values(type, 16385);
        input = parent.slice(0, 0, 16384);
        output = parent.slice(0, 1, 16385);
        float[] expected = expected(input, type);
        CpuNdElementwise.unaryInto(input, output, CpuCompactUnaryTest::equation);
        assertArrayEquals(expected, output.toFloatArray());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void geluAndItsGradientsAreIndependentOfContiguousLayout(DType type) {
        Tensor dense = values(type, 16384).reshape(128, 128).requiresGrad(true);
        Tensor strided = new Tensor(dense).transpose().requiresGrad(true);
        Tensor a = dense.geluActivation(), b = strided.geluActivation();
        assertArrayEquals(a.toFloatArray(), b.transpose().toFloatArray());
        a.sumTensor().backward();
        b.sumTensor().backward();
        assertArrayEquals(dense.grad().toFloatArray(), strided.grad().transpose().toFloatArray());
    }

    @ParameterizedTest @EnumSource(DType.class)
    void castsMatchScalarConversionsForEveryStoragePair(DType source) {
        Tensor input = values(source, 65539).slice(0, 2, 65538);
        for (DType destination : DType.values()) {
            Tensor expected = Tensor.from(input.toFloatArray(), destination, input.numel());
            Tensor actual = CpuNdElementwise.cast(input, destination);
            assertEquals(destination, actual.dtype());
            assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
            Tensor strided = input.reshape(256, 256).transpose();
            assertArrayEquals(actual.reshape(256, 256).transpose().toFloatArray(),
                    CpuNdElementwise.cast(strided, destination).toFloatArray());
        }
    }

    @ParameterizedTest @EnumSource(DType.class)
    void deferredHostStorageIsInitializedBeforeWorkersWrite(DType type) {
        Tensor input = Tensor.deviceOnly(type, 65539), output = Tensor.deviceOnly(type, 65539);
        CpuNdElementwise.unaryInto(input, output, value -> value + 3);
        float[] expected = new float[65539];
        java.util.Arrays.fill(expected, 3);
        assertArrayEquals(expected, output.toFloatArray());
        output = Tensor.deviceOnly(type, 65539);
        Tensor scalar = Tensor.from(new float[]{3}, type, new int[0]);
        CpuNdElementwise.binaryInto(input, scalar, output, Float::sum);
        assertArrayEquals(expected, output.toFloatArray());
    }

    private static float[] expected(Tensor input, DType type) {
        float[] values = input.toFloatArray();
        for (int index = 0; index < values.length; index++) { values[index] = equation(values[index]); }
        return Tensor.from(values, type, values.length).toFloatArray();
    }

    private static float equation(float value) { return 0.5f * value * (1 + (float) Math.tanh(value)); }

    private static Tensor values(DType type, int count) {
        float[] values = new float[count];
        for (int index = 0; index < count; index++) { values[index] = (index % 129 - 64) * 0.125f; }
        float[] special = {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
            -0f, 0f, Float.MIN_VALUE, -Float.MIN_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE};
        System.arraycopy(special, 0, values, 0, Math.min(special.length, count));
        return Tensor.from(values, type, count);
    }
}
