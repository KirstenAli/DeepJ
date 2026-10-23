package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;
import java.util.Random;

public final class BiasFreeLinear implements Projection {

    private final Parameter weight;

    public BiasFreeLinear(int inputs, int outputs, Random random) {
        if (inputs <= 0 || outputs <= 0) throw new IllegalArgumentException("Dimensions must be positive");
        weight = new Parameter(Tensor.random(inputs, outputs, random));
    }

    @Override
    public Tensor forward(Tensor value) {
        return value.matmul(weight.value);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weight);
    }
}
