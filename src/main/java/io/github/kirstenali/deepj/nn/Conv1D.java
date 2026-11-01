package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.List;
import java.util.Random;

/** Temporal cross-correlation for [batch, channels, length], sharing the 2D implementation. */
public final class Conv1D extends Module {
    private final Conv2D convolution;

    public Conv1D(int inputs, int outputs, int kernel, Random random) {
        this(inputs, outputs, kernel, 1, 0, 1, true, random, ParameterFactory.float32());
    }

    public Conv1D(int inputs, int outputs, int kernel, int stride, int padding, int dilation,
                  boolean bias, Random random, ParameterFactory parameters) {
        var window = new Window2D(1, kernel, 1, stride, 0, padding, 1, dilation);
        convolution = new Conv2D(inputs, outputs, window, bias, random, parameters);
    }

    public Parameter weight() { return convolution.weight(); }
    public Parameter bias() { return convolution.bias(); }
    @Override protected List<Layer> children() { return List.of(convolution); }

    @Override public Tensor forward(Tensor input) {
        NeuralChecks.rank(input, 3);
        try (var scope = TensorScope.open()) {
            return scope.escape(convolution.forward(input.unsqueeze(2)).select(2, 0));
        }
    }
}
