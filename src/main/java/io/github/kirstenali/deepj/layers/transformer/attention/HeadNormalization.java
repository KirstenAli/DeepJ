package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.layers.transformer.norm.RMSNorm1D;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;

final class HeadNormalization implements Layer {

    private final RMSNorm1D norm;

    HeadNormalization(int dimension, boolean enabled) {
        norm = enabled ? new RMSNorm1D(dimension) : null;
    }

    @Override
    public Tensor forward(Tensor input) {
        return norm == null ? input : norm.forward(input);
    }

    @Override
    public Tensor backward(Tensor gradient) {
        return norm == null ? gradient : norm.backward(gradient);
    }

    @Override
    public List<Parameter> parameters() {
        return norm == null ? List.of() : norm.parameters();
    }
}
