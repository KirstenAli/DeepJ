package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalCompactImportTest extends MetalDifferentialTestSupport {
    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void repeatedImportsReadUpdatedBuffersAndRespectLeadingViewShapes(DType dtype) {
        try (var ignored = GradMode.noGrad()) {
            Tensor left = Tensor.ones(dtype, 64, 256), right = Tensor.zeros(dtype, 256, 256);
            check(left, right, 0);
            metal.copyInto(right, Tensor.ones(dtype, 256, 256));
            check(left, right, 256);
            check(left.slice(0, 0, 1), right, 256);
            metal.copyInto(left, Tensor.ones(dtype, 64, 256).multiplyScalar(.5f));
            check(left.slice(0, 0, 1), right, 128);
            check(left, right, 128);
        }
    }

    @ParameterizedTest @EnumSource(value = DType.class, names = {"FLOAT16", "BFLOAT16"})
    void longCompactStreamsPreserveDependenciesAndFloat32Gradients(DType dtype) {
        float[] identity = new float[256 * 256];
        for (int index = 0; index < 256; index++) { identity[index * 257] = 1; }
        Tensor input = Tensor.ones(dtype, 64, 256).requiresGrad(true);
        Tensor weight = Tensor.from(identity, dtype, 256, 256).requiresGrad(true), output = input;
        for (int step = 0; step < 20; step++) { output = output.matmul(weight).addScalar(1); }
        constant(output, 21);
        output.sumTensor().backward();
        assertEquals(DType.FLOAT32, input.grad().dtype());
        assertEquals(DType.FLOAT32, weight.grad().dtype());
        constant(input.grad(), 1);
        constant(weight.grad(), 13440);
    }

    private static void constant(Tensor value, float expected) {
        for (float actual : value.toFloatArray()) { assertEquals(expected, actual); }
    }

    private void check(Tensor left, Tensor right, float expected) {
        Tensor result = metal.matmul(left, right);
        assertArrayEquals(new int[]{left.matrixRows(), right.matrixCols()}, result.shape());
        for (float actual : result.toFloatArray()) { assertEquals(expected, actual); }
    }
}
