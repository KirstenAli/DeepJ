package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Random;
import static io.github.kirstenali.deepj.nn.NeuralTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class RecurrentTest extends NeuralTest {
    @Test void cellsMatchIndependentScalarEquations() {
        for (RecurrentCell cell : cells()) {
            initialize(cell);
            Tensor input = values(2, 2), hidden = values(2, 2).addScalar(0.3f);
            Tensor memory = cell instanceof LSTMCell ? values(2, 2).addScalar(0.5f) : null;
            RecurrentState output = cell.step(input, new RecurrentState(hidden, memory));
            for (int sample = 0; sample < 2; sample++) { compare(cell, input, hidden, memory, output, sample); }
        }
    }

    @Test void stateAndParameterGradientsMatchFiniteDifferences() {
        for (RecurrentCell cell : cells()) {
            initialize(cell);
            Tensor hidden = values(2, 2).requiresGrad(true);
            Tensor memory = cell instanceof LSTMCell ? values(2, 2).requiresGrad(true) : null;
            Module adapted = stateful(cell, new RecurrentState(hidden, memory));
            if (memory == null) gradients(adapted, values(2, 2), hidden);
            else gradients(adapted, values(2, 2), hidden, memory);
        }
    }

    @Test void recurrentSequencesSupportMaskedLengthsAndExplicitChunkedState() {
        for (RecurrentCell cell : cells()) {
            Recurrent network = new Recurrent(cell);
            initialize(network);
            Tensor input = values(2, 3, 2).requiresGrad(true);
            RecurrentResult masked = network.run(input, null, new int[]{3, 1});
            assertArrayEquals(new int[]{2, 3, 2}, masked.sequence().shape());
            assertArrayEquals(new float[]{0, 0}, masked.sequence().select(0, 1).select(0, 2).toFloatArray(), 0.0f);
            masked.sequence().sumTensor().backward();
            assertArrayEquals(new float[]{0, 0}, input.grad().select(0, 1).select(0, 2).toFloatArray(), 0.0f);
            checkChunks(network, input.detach());
            gradients(network, values(1, 2, 2));
        }
    }

    @Test void zeroLengthSamplesKeepInitialStateAndDetachedStatesHaveIndependentStorage() {
        Recurrent network = new LSTM(2, 2, new Random(1));
        RecurrentState initial = new RecurrentState(values(2, 2), values(2, 2));
        RecurrentResult result = network.run(values(2, 2, 2), initial, new int[]{0, 2});
        assertArrayEquals(initial.hidden().select(0, 0).toFloatArray(),
                result.state().hidden().select(0, 0).toFloatArray());
        assertArrayEquals(initial.cell().select(0, 0).toFloatArray(),
                result.state().cell().select(0, 0).toFloatArray());
        RecurrentState detached = result.state().detach();
        assertFalse(detached.hidden().requiresGrad());
        set(detached.hidden(), 0, 900);
        assertNotEquals(900, result.state().hidden().get(0, 0));
        assertNull(new RecurrentState(values(1, 2)).detach().cell());
    }

    @Test void compactCellsAndConvenienceNetworksShareTheExistingParameterPolicy() {
        for (DType dtype : DType.values()) {
            ParameterFactory factory = new ParameterFactory(dtype);
            List<Recurrent> networks = List.of(new RNN(2, 2, new Random(1), factory),
                    new GRU(2, 2, new Random(2), factory), new LSTM(2, 2, new Random(3), factory));
            for (Recurrent network : networks) {
                Tensor output = network.forward(values(1, 2, 2).to(dtype));
                assertArrayEquals(new int[]{1, 2, 2}, output.shape());
                output.sumTensor().backward();
                for (Parameter parameter : network.parameters()) { assertNotNull(parameter.grad()); }
            }
        }
        assertEquals(2, new RNN(2, 2, new Random(1)).cell().inputSize());
        assertEquals(2, new GRU(2, 2, new Random(1)).cell().hiddenSize());
        assertArrayEquals(new int[]{1, 2}, new RNNCell(2, 2, new Random(1)).forward(values(1, 2)).shape());
    }

    private static List<RecurrentCell> cells() {
        return List.of(new RNNCell(2, 2, new Random(1)), new GRUCell(2, 2, new Random(2)),
                new LSTMCell(2, 2, new Random(3)));
    }

    private static Module stateful(RecurrentCell cell, RecurrentState state) {
        return new Module() {
            @Override public Tensor forward(Tensor input) { return cell.step(input, state).hidden(); }
            @Override protected List<Layer> children() { return List.of(cell); }
        };
    }

    private static void checkChunks(Recurrent network, Tensor input) {
        RecurrentResult first = network.run(input.slice(1, 0, 1), null, null);
        RecurrentResult second = network.run(input.slice(1, 1, 3), first.state(), null);
        Tensor joined = TensorJoin.concatenate(1, first.sequence(), second.sequence());
        assertArrayEquals(network.forward(input).toFloatArray(), joined.toFloatArray(), 1e-6f);
    }

    private static void compare(RecurrentCell cell, Tensor input, Tensor hidden, Tensor memory,
                                RecurrentState result, int sample) {
        float[] x = projection(cell.parameters().subList(0, 2), input, sample);
        float[] h = projection(cell.parameters().subList(2, 4), hidden, sample);
        for (int index = 0; index < cell.hiddenSize(); index++) {
            double[] expected = scalar(cell, x, h, hidden.get(sample, index),
                    memory == null ? 0 : memory.get(sample, index), index);
            assertEquals(expected[0], result.hidden().get(sample, index), 1e-6);
            if (memory != null) assertEquals(expected[1], result.cell().get(sample, index), 1e-6);
        }
    }

    private static float[] projection(List<Parameter> parameters, Tensor input, int sample) {
        Tensor weights = parameters.get(0).value, bias = parameters.get(1).value;
        float[] values = new float[weights.size(1)];
        for (int output = 0; output < values.length; output++) {
            values[output] = bias.get(0, output);
            for (int index = 0; index < input.size(1); index++) {
                values[output] += input.get(sample, index) * weights.get(index, output);
            }
        }
        return values;
    }

    private static double[] scalar(RecurrentCell cell, float[] x, float[] h, float old, float memory, int index) {
        int width = cell.hiddenSize();
        if (cell instanceof RNNCell) return new double[]{Math.tanh(x[index] + h[index])};
        if (cell instanceof GRUCell) {
            double reset = sigmoid(x[index] + h[index]), update = sigmoid(x[width + index] + h[width + index]);
            double candidate = Math.tanh(x[2 * width + index] + reset * h[2 * width + index]);
            return new double[]{(1 - update) * candidate + update * old};
        }
        double written = sigmoid(x[index] + h[index]) * Math.tanh(x[2 * width + index] + h[2 * width + index]);
        double next = sigmoid(x[width + index] + h[width + index]) * memory + written;
        return new double[]{sigmoid(x[3 * width + index] + h[3 * width + index]) * Math.tanh(next), next};
    }

    private static double sigmoid(double value) { return 1 / (1 + Math.exp(-value)); }
}
