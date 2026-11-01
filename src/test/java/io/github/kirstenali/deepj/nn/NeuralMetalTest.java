package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.activations.Tanh;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Stream;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class NeuralMetalTest extends NeuralTest {
    @ParameterizedTest(name = "{0}, {1}")
    @MethodSource("cases")
    void selectedMetalBackendMatchesCpuForwardAndBackward(Case specification, DType dtype) {
        assumeTrue(MetalBackend.isAvailable(), "Metal device unavailable; this is not a GPU pass");
        Result cpu = execute(specification, dtype, false);
        Result metal = execute(specification, dtype, true);
        float tolerance = dtype == DType.FLOAT32 ? 0.0002f : dtype == DType.FLOAT16 ? 0.003f : 0.02f;
        assertArrayEquals(cpu.output(), metal.output(), tolerance);
        assertArrayEquals(cpu.inputGradient(), metal.inputGradient(), tolerance);
        for (int index = 0; index < cpu.gradients().size(); index++) {
            assertArrayEquals(cpu.gradients().get(index), metal.gradients().get(index), tolerance);
        }
        for (int index = 0; index < cpu.buffers().size(); index++) {
            assertArrayEquals(cpu.buffers().get(index), metal.buffers().get(index), tolerance);
        }
    }

    private static Stream<Arguments> cases() {
        return specifications().stream().flatMap(specification -> Arrays.stream(DType.values())
                .map(dtype -> Arguments.of(specification, dtype)));
    }

    private static List<Case> specifications() {
        return List.of(new Case("FNN", p -> new FNN(2, new int[]{3}, 2, Tanh::new, null, new Random(1), p), 2, 2),
                new Case("Conv2D", p -> new Conv2D(2, 2, new Window2D(2, 2, 1, 1, 1, 1, 1, 1),
                        true, new Random(2), p), 2, 2, 3, 3),
                new Case("Conv1D", p -> new Conv1D(2, 2, 2, 1, 1, 2, true, new Random(2), p), 2, 2, 4),
                new Case("MaxPool", p -> Pool2D.max(2), 2, 2, 4, 4),
                new Case("AveragePool", p -> Pool1D.average(2), 2, 2, 4),
                new Case("BatchNorm", p -> new BatchNorm(2, 1e-5f, 0.1f, p), 2, 2, 3),
                new Case("Dropout", p -> new Dropout(0.4f, 42), 2, 3),
                new Case("RNN", p -> new RNN(2, 2, new Random(1), p), 2, 3, 2),
                new Case("GRU", p -> new GRU(2, 2, new Random(1), p), 2, 3, 2),
                new Case("LSTM", p -> new LSTM(2, 2, new Random(1), p), 2, 3, 2),
                new Case("Encoder", p -> new TransformerEncoder(new TransformerEncoderConfig(4, 2, 6, 1, 0),
                        new Random(1), p), 2, 3, 4),
                new Case("Autoencoder", p -> new Autoencoder(new FNN(2, new int[0], 1, Tanh::new, null,
                        new Random(1), p.scope("encoder")), new FNN(1, new int[0], 2, Tanh::new, null,
                        new Random(2), p.scope("decoder"))), 2, 2));
    }

    private static Result execute(Case specification, DType dtype, boolean metal) {
        Tensor.setBackend(metal ? new MetalBackend() : new CpuBackend());
        try {
            Layer model = specification.create().apply(new ParameterFactory(dtype));
            initialize(model);
            Tensor input = values(specification.shape()).to(dtype).requiresGrad(true), output = model.forward(input);
            if (metal) assertNotNull(output.getGpuTag(), "Output must remain on Metal");
            output.multiply(values(output.shape()).addScalar(0.2f)).sumTensor().backward();
            return new Result(output.toFloatArray(), input.grad().toFloatArray(),
                    model.parameters().stream().map(parameter -> parameter.grad().toFloatArray()).toList(),
                    buffers(model).stream().map(Tensor::toFloatArray).toList());
        } finally {
            Tensor.backend().releaseResources();
            Tensor.setBackend(new CpuBackend());
        }
    }

    private static List<Tensor> buffers(Layer model) {
        return model instanceof Module module ? module.buffers() : List.of();
    }

    private record Case(String name, Function<ParameterFactory, Layer> create, int... shape) {
        @Override public String toString() { return name; }
    }
    private record Result(float[] output, float[] inputGradient, List<float[]> gradients, List<float[]> buffers) {}
}
