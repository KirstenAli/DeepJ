package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/** Cross-correlation of [batch, channels, height, width] inputs, using gather and matrix multiplication. */
public final class Conv2D extends Module {
    private final int inputs;
    private final int outputs;
    private final Window2D window;
    private final Parameter weight;
    private final Parameter bias;

    public Conv2D(int inputs, int outputs, int kernel, Random random) {
        this(inputs, outputs, new Window2D(kernel, kernel), true, random, ParameterFactory.float32());
    }

    public Conv2D(int inputs, int outputs, Window2D window, boolean bias, Random random,
                  ParameterFactory parameters) {
        this.inputs = NeuralChecks.positive(inputs, "input channels");
        this.outputs = NeuralChecks.positive(outputs, "output channels");
        this.window = Objects.requireNonNull(window, "window");
        weight = parameters.random("weight", Math.multiplyExact(inputs, window.area()), outputs, random);
        this.bias = bias ? parameters.zeros("bias", 1, outputs) : null;
    }

    public Parameter weight() { return weight; }
    public Parameter bias() { return bias; }
    public Window2D window() { return window; }
    @Override public List<Parameter> parameters() { return bias == null ? List.of(weight) : List.of(weight, bias); }

    @Override public Tensor forward(Tensor input) {
        NeuralChecks.rank(input, 4);
        if (input.size(1) != inputs) throw new IllegalArgumentException("Convolution input channel count differs");
        try (var scope = TensorScope.open()) {
            SpatialPatches patches = SpatialPatches.gather(input, window, 0);
            Tensor columns = patches.values().permute(0, 2, 1, 3)
                    .reshape(input.size(0), patches.height() * patches.width(), inputs * window.area());
            Tensor result = columns.matmul(weight.value);
            if (bias != null) result = result.add(bias.value);
            return scope.escape(result.reshape(input.size(0), patches.height(), patches.width(), outputs)
                    .permute(0, 3, 1, 2));
        }
    }
}
