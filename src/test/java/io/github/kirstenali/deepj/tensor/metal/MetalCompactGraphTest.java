package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MetalCompactGraphTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void compactGraphsRespectOffsetsPaddingAndBothTransposes(DType type) {
        for (int rows : new int[]{1, 2, 17, 33, 63, 64, 256}) {
            Tensor left = values(type, rows + 2, 259).slice(0, 1, rows + 1).slice(1, 1, 258);
            Tensor transposed = values(type, 257, rows).transpose(-2, -1);
            Tensor right = values(type, 259, 261).slice(0, 1, 258).slice(1, 1, 260);
            Tensor other = values(type, 259, 257).transpose(-2, -1);
            compare(left, right);
            compare(left, other);
            compare(transposed, right);
            compare(transposed, other);
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void graphAndCustomKernelsShareOneStreamAndKeepFloat32Gradients(DType type) {
        Tensor input = values(type, 64, 256).requiresGrad(true);
        Tensor weight = identity(type).requiresGrad(true);
        Tensor result = input.matmul(weight).multiplyScalar(2).matmul(weight);
        float[] expected = input.toFloatArray();
        for (int index = 0; index < expected.length; index++) { expected[index] *= 2; }
        assertArrayEquals(expected, result.toFloatArray());
        result.sumTensor().backward();
        Arrays.fill(expected, 2);
        assertEquals(DType.FLOAT32, input.grad().dtype());
        assertEquals(DType.FLOAT32, weight.grad().dtype());
        assertArrayEquals(expected, input.grad().toFloatArray());
        assertArrayEquals(expected, input.matmul(weight).multiplyScalar(0).addScalar(2).toFloatArray());
    }

    private void compare(Tensor left, Tensor right) {
        Tensor expected = cpu.matmul(left, right), actual = left.matmul(right);
        assertEquals(left.dtype(), actual.dtype());
        float tolerance = left.dtype() == DType.BFLOAT16 ? .008f : .001f;
        assertTensorClose(expected, actual, .00001f, tolerance);
    }

    private static Tensor identity(DType type) {
        float[] values = new float[256 * 256];
        for (int index = 0; index < 256; index++) { values[index * 257] = 1; }
        return Tensor.from(values, type, 256, 256);
    }

    private static Tensor values(DType dtype, int rows, int columns) {
        float[] data = new float[rows * columns];
        var random = new Random(713 + rows + columns);
        for (int index = 0; index < data.length; index++) { data[index] = random.nextFloat() * .5f - .25f; }
        return Tensor.from(data, dtype, rows, columns);
    }
}
