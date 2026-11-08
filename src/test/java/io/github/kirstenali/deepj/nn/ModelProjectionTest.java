package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.util.Random;
import java.util.function.BiFunction;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class ModelProjectionTest {
    @AfterEach void restore() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(new CpuBackend());
    }

    private static void backend(String name) {
        if (name.equals("metal")) Assumptions.assumeTrue(MetalBackend.isAvailable());
        Tensor.setBackend(name.equals("metal") ? new MetalBackend() : new CpuBackend());
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void compactInferenceProjectsIndependentBatchRowsLikeAFlatMatrix(String name) {
        backend(name);
        try (var ignored = GradMode.noGrad()) {
            Tensor input = Tensor.random(256, 512, new Random(41)).to(DType.BFLOAT16).reshape(2, 128, 512);
            Tensor weight = Tensor.random(512, 512, new Random(42)).to(DType.BFLOAT16);
            float[] expected = input.reshape(256, 512).matmul(weight).toFloatArray();
            assertArrayEquals(expected, ModelOps.project(input, weight).toFloatArray(), 0);
        }
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void viewsAndOtherPrecisionsKeepTheirExistingProductPath(String name) {
        backend(name);
        try (var ignored = GradMode.noGrad()) {
            Tensor input = Tensor.random(128, 256, new Random(8)).reshape(2, 64, 256);
            Tensor weight = Tensor.random(256, 256, new Random(9));
            direct(input, weight);
            direct(input.to(DType.FLOAT16), weight.to(DType.FLOAT16));
            Tensor compact = Tensor.random(192, 256, new Random(10)).to(DType.BFLOAT16);
            Tensor matrix = weight.to(DType.BFLOAT16);
            direct(compact.slice(0, 0, 128).reshape(64, 2, 256).transpose(0, 1), matrix);
            direct(compact.reshape(3, 64, 256).slice(0, 1, 3), matrix);
        }
    }

    private static void direct(Tensor input, Tensor weight) {
        assertArrayEquals(input.matmul(weight).toFloatArray(), ModelOps.project(input, weight).toFloatArray(), 0);
    }

    @ParameterizedTest @ValueSource(strings = {"cpu", "metal"})
    void trainingKeepsTheOriginalBatchedProductsAndGradients(String name) {
        backend(name);
        float[][] expected = gradients(Tensor::matmul), actual = gradients(ModelOps::project);
        for (int index = 0; index < expected.length; index++) {
            assertArrayEquals(expected[index], actual[index], 0);
        }
    }

    private static float[][] gradients(BiFunction<Tensor, Tensor, Tensor> product) {
        Tensor input = Tensor.random(128, 256, new Random(13)).to(DType.BFLOAT16)
                .reshape(2, 64, 256).requiresGrad(true);
        Tensor weight = Tensor.random(256, 256, new Random(14)).to(DType.BFLOAT16).requiresGrad(true);
        Tensor output = product.apply(input, weight);
        float[] values = output.toFloatArray();
        output.sumTensor().backward();
        return new float[][]{values, input.grad().toFloatArray(), weight.grad().toFloatArray()};
    }
}
