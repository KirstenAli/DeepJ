package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** FLOAT32 results of compact-operand ops must stay eligible for the fused dense kernels. */
class MetalCompactOutputDensityTest extends MetalDifferentialTestSupport {
    @Test
    void float32ProductOfCompactWeightCoversStorage() {
        Tensor result = metal.matmul(random(64, 128, 1), compact(DType.FLOAT16, 128, 96, 2));
        assertDense(result, "matmul");
    }

    @Test
    void float32SumWithCompactBiasCoversStorage() {
        Tensor bias = compact(DType.FLOAT16, 1, 96, 3).reshape(96);
        assertDense(metal.add(random(64, 96, 1), bias), "add");
    }

    @Test
    void compactEmbeddingLookupsWidenToFloat32() {
        var factory = new io.github.kirstenali.deepj.optimisers.ParameterFactory(DType.FLOAT16);
        var embedding = new io.github.kirstenali.deepj.transformer.embeddings.Embedding(
                16, 64, new java.util.Random(1), factory);
        assertEquals(DType.FLOAT16, embedding.weight().value.dtype());
        assertDense(embedding.forward(new int[]{1, 5, 9}), "embedding");
    }

    @Test
    void compactFactoriesKeepRowVectorsFloat32() {
        var factory = new io.github.kirstenali.deepj.optimisers.ParameterFactory(DType.BFLOAT16);
        assertEquals(DType.FLOAT32, factory.ones("gamma", 1, 8).value.dtype());
        assertEquals(DType.FLOAT32, factory.zeros("bias", 1, 8).value.dtype());
        assertEquals(DType.BFLOAT16, factory.zeros("weight", 8, 8).value.dtype());
    }

    private static void assertDense(Tensor result, String label) {
        assertTrue(MetalDenseOps.matrix(result),
                label + " dtype=" + result.dtype() + " rank=" + result.rank() + " covers=" + result.coversStorage());
    }

    private Tensor compact(DType dtype, int rows, int columns, long seed) {
        return Tensor.from(random(rows, columns, seed).toFloatArray(), dtype, rows, columns);
    }
}

