package io.github.kirstenali.deepj.layers;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;
import java.util.Random;

public final class BiasFreeLinear implements Projection {

    private final Parameter weight;

    public BiasFreeLinear(int inputs, int outputs, Random random) {
        this(inputs, outputs, random, ParameterFactory.float32());
    }

    public BiasFreeLinear(int inputs, int outputs, Random random, ParameterFactory parameters) {
        if (inputs <= 0 || outputs <= 0) throw new IllegalArgumentException("Dimensions must be positive");
        weight = parameters.random("weight", inputs, outputs, random);
    }

    @Override
    public Tensor forward(Tensor value) {
        return ProjectionOps.linear(value, weight.value);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weight);
    }
}
