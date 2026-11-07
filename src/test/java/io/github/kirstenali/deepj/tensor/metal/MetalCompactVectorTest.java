package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalCompactVectorTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(DType.class)
    void compactVectorsHandleBatchesOffsetsTransposesAndPartialTiles(DType leftType) {
        for (DType rightType : DType.values()) {
            Tensor left = values(leftType, 2, 1, 259).slice(2, 1, 258);
            Tensor right = values(rightType, 1, 257, 261).slice(2, 1, 260);
            assertProduct(left, right);
            assertProduct(left, values(rightType, 1, 259, 257).transpose(-2, -1));
            assertProduct(left, values(rightType, 1, 257, 518).slice(2, 0, 518, 2));
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void mixedVectorsNeverNarrowFloat32ValuesAndPreserveGradients(DType type) {
        float[] input = new float[256], identity = new float[256 * 256];
        Arrays.fill(input, 131072 + .015625f);
        for (int index = 0; index < 256; index++) { identity[index * 257] = 1; }
        Tensor left = Tensor.from(input, 1, 256).requiresGrad(true);
        Tensor right = Tensor.from(identity, type, 256, 256).requiresGrad(true);
        Tensor output = left.matmul(right);
        assertEquals(DType.FLOAT32, output.dtype());
        assertArrayEquals(input, output.toFloatArray());
        output.sumTensor().backward();
        assertIdentityGradients(left, right, input[0]);
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactVectorProjectionsPreserveFloat32Gradients(DType dtype) {
        float[] identity = new float[256 * 256];
        for (int index = 0; index < 256; index++) { identity[index * 257] = 1; }
        Tensor left = Tensor.ones(dtype, 1, 256).requiresGrad(true);
        Tensor right = Tensor.from(identity, dtype, 256, 256).requiresGrad(true);
        Tensor output = left.matmul(right);
        assertArrayEquals(left.toFloatArray(), output.toFloatArray());
        output.sumTensor().backward();
        assertIdentityGradients(left, right, 1);
    }

    private static void assertIdentityGradients(Tensor left, Tensor right, float expected) {
        assertEquals(DType.FLOAT32, left.grad().dtype());
        assertEquals(DType.FLOAT32, right.grad().dtype());
        float[] ones = new float[256];
        Arrays.fill(ones, 1);
        assertArrayEquals(ones, left.grad().toFloatArray());
        for (float value : right.grad().toFloatArray()) { assertEquals(expected, value); }
    }

    @Test void halfSubnormalsAccumulateBeforeOutputRounding() {
        float[] input = new float[256], weights = new float[256 * 257], expected = new float[257];
        input[0] = Math.scalb(1.0f, -24);
        input[1] = input[0];
        Arrays.fill(weights, 0, 514, .5f);
        Arrays.fill(expected, input[0]);
        Tensor left = Tensor.from(input, DType.FLOAT16, 1, 256);
        Tensor right = Tensor.from(weights, DType.FLOAT16, 256, 257);
        assertArrayEquals(expected, left.matmul(right).toFloatArray());
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void infinitiesAndNanAreNotClipped(DType type) {
        float[] input = new float[256], weights = new float[256 * 259], expected = new float[259];
        Arrays.fill(weights, 1);
        Tensor right = Tensor.from(weights, type, 256, 259);
        for (float special : new float[]{Float.POSITIVE_INFINITY, Float.NaN}) {
            input[0] = special;
            Arrays.fill(expected, special);
            assertArrayEquals(expected, Tensor.from(input, type, 1, 256).matmul(right).toFloatArray());
        }
    }

    private void assertProduct(Tensor left, Tensor right) {
        float[] a = left.toFloatArray(), b = right.toFloatArray();
        int inner = left.size(-1), columns = right.size(-1);
        float[] expected = new float[left.size(0) * columns];
        for (int batch = 0; batch < left.size(0); batch++) {
            for (int column = 0; column < columns; column++) {
                expected[batch * columns + column] = dot(a, b, batch * inner, inner, column, columns);
            }
        }
        Tensor actual = left.matmul(right);
        DType dtype = DType.promote(left.dtype(), right.dtype());
        assertEquals(dtype, actual.dtype());
        float tolerance = dtype == DType.BFLOAT16 ? .008f : dtype == DType.FLOAT16 ? .001f : .00001f;
        assertTensorClose(Tensor.from(expected, dtype, actual.shape()), actual, .00001f, tolerance);
    }

    private static float dot(float[] a, float[] b, int offset, int count, int column, int columns) {
        double sum = 0;
        for (int inner = 0; inner < count; inner++) { sum += (double) a[offset + inner] * b[inner * columns + column]; }
        return (float) sum;
    }

    private static Tensor values(DType dtype, int... shape) {
        int count = Arrays.stream(shape).reduce(1, Math::multiplyExact);
        float[] data = new float[count];
        var random = new Random(213 + count);
        for (int index = 0; index < count; index++) { data[index] = random.nextFloat() * .5f - .25f; }
        return Tensor.from(data, dtype, shape);
    }
}
