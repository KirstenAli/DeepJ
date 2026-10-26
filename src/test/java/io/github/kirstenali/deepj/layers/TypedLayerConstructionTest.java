package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.layers.transformer.norm.*;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.transformer.embeddings.*;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class TypedLayerConstructionTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @EnumSource(DType.class)
    void leafLayersConstructCompactParametersAndDifferentiateWithoutPrototypeWeights(DType dtype) {
        ParameterFactory factory = new ParameterFactory(dtype);
        Embedding tokens = new Embedding(8, 4, new Random(1), factory.scope("tokens"));
        PositionalEmbedding positions = new PositionalEmbedding(4, 4, new Random(2), factory.scope("positions"));
        Linear linear = new Linear(4, 4, new Random(3), factory.scope("linear"));
        BiasFreeLinear projection = new BiasFreeLinear(4, 2, new Random(4), factory.scope("projection"));
        LayerNorm1D layerNorm = new LayerNorm1D(4, factory.scope("layerNorm"));
        RMSNorm1D rmsNorm = new RMSNorm1D(4, factory.scope("rmsNorm"));
        Tensor input = tokens.forward(new int[]{1, 3}).add(positions.forward(2));
        Tensor output = projection.forward(rmsNorm.forward(layerNorm.forward(linear.forward(input))));
        assertEquals(dtype, output.dtype());
        output.sumTensor().backward();
        assertParameters(dtype, List.of(tokens, positions, linear, projection, layerNorm, rmsNorm));
    }

    private static void assertParameters(DType dtype, List<io.github.kirstenali.deepj.training.Trainable> layers) {
        for (var layer : layers) {
            for (var parameter : layer.parameters()) {
                assertEquals(dtype, parameter.value.dtype());
                assertEquals(dtype.storageBytes(parameter.value.numel()), parameter.value.storageBytes());
                assertEquals(DType.FLOAT32, parameter.grad().dtype());
                assertTrue(parameter.name().contains("."));
                assertFinite(parameter.grad());
            }
        }
    }

    private static void assertFinite(Tensor gradient) {
        for (float value : gradient.toFloatArray()) {
            assertTrue(Float.isFinite(value));
        }
    }
}
