package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class CpuCompactPointwiseTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void nativeTilesMatchScalarArithmeticAcrossAllDtypePairs(DType leftType) {
        for (DType rightType : DType.values()) {
            for (int count : new int[]{255, 256, 257, 8192, 8193, 65536}) {
                Tensor left = values(leftType, count + 3).slice(0, 1, count + 1);
                Tensor right = values(rightType, count + 7).slice(0, 2, count + 2);
                compare(left, right);
            }
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void specialValuesAndAliasedAdditionKeepTheirFallbackSemantics(DType type) {
        Tensor left = values(type, 8193), right = new Tensor(left);
        float[] special = {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -100, 100, 0};
        for (int index = 0; index < special.length; index++) { left.memory().set(index, special[index]); }
        compare(left, right);
        float[] expected = CpuNdElementwise.binary(left, left, Float::sum).toFloatArray();
        Tensor.backend().addInPlace(left, left);
        assertArrayEquals(expected, left.toFloatArray());
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactBackwardRetainsExactFloat32Adjoints(DType type) {
        for (int count : new int[]{256, 8193, 65536}) {
            Tensor gate = values(type, count), up = values(type, count), gradient = values(DType.FLOAT32, count);
            float[] special = {Float.NaN, Float.NEGATIVE_INFINITY, -100, 100};
            for (int index = 0; index < special.length; index++) { gate.memory().set(index, special[index]); }
            var expected = new SwiGluBackwardResult(new Tensor(count), new Tensor(count));
            CpuNdSwiGlu.backwardRange(gradient, gate, up, expected, 0, count);
            var actual = CpuNdSwiGlu.backward(gradient, gate, up);
            assertArrayEquals(expected.gateGradient().toFloatArray(), actual.gateGradient().toFloatArray());
            assertArrayEquals(expected.upGradient().toFloatArray(), actual.upGradient().toFloatArray());
        }
    }

    private static void compare(Tensor left, Tensor right) {
        float[] a = left.toFloatArray(), b = right.toFloatArray();
        float[] sum = new float[a.length], activated = new float[a.length];
        for (int index = 0; index < a.length; index++) {
            sum[index] = a[index] + b[index];
            activated[index] = a[index] * (1 / (1 + (float) Math.exp(-a[index]))) * b[index];
        }
        DType type = DType.promote(left.dtype(), right.dtype());
        assertArrayEquals(Tensor.from(sum, type, a.length).toFloatArray(),
                Tensor.backend().add(left, right).toFloatArray());
        assertArrayEquals(Tensor.from(activated, type, a.length).toFloatArray(),
                CpuNdSwiGlu.apply(left, right).toFloatArray());
    }

    private static Tensor values(DType type, int count) {
        float[] values = new float[count];
        var random = new Random(385 + count);
        for (int index = 0; index < count; index++) { values[index] = (float) random.nextGaussian(); }
        return Tensor.from(values, type, count);
    }
}
