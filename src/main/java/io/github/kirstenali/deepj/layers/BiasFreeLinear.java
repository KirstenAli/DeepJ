package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;
import java.util.Random;

public final class BiasFreeLinear implements Projection {

    private final Parameter weight;
    private Tensor input;

    public BiasFreeLinear(int inputs, int outputs, Random random) {
        if (inputs <= 0 || outputs <= 0) throw new IllegalArgumentException("Dimensions must be positive");
        weight = new Parameter(Tensor.random(inputs, outputs, random));
    }

    @Override
    public Tensor forward(Tensor value) {
        input = value;
        return value.matmul(weight.value);
    }

    @Override
    public Tensor backward(Tensor gradient) {
        weight.grad.addInPlace(Tensor.backend().batchedMatmul(input, gradient, 1, true, false));
        return Tensor.backend().batchedMatmul(gradient, weight.value, 1, false, true);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weight);
    }
}
