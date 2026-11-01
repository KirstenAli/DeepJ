package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.activations.ActivationFunction;
import io.github.kirstenali.deepj.activations.ReLU;
import io.github.kirstenali.deepj.layers.FNN;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/** Multilayer perceptron backed by DeepJ's existing FNN implementation. */
public final class MLP extends Module {
    private final FNN network;

    public MLP(int inputs, int[] hidden, int outputs, Random random) {
        this(inputs, hidden, outputs, ReLU::new, null, random, ParameterFactory.float32());
    }

    public MLP(int inputs, int[] hidden, int outputs, Supplier<ActivationFunction> activation,
               ActivationFunction output, Random random, ParameterFactory parameters) {
        network = new FNN(inputs, hidden, outputs, activation, output, random, parameters);
    }

    @Override public Tensor forward(Tensor input) { return network.forward(input); }
    @Override protected List<Layer> children() { return List.of(network); }
}
