package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.autograd.NormalizationOps;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;

public final class LayerNorm1D implements NormLayer {
    private static final float EPS = 1e-5f;
    private final int dim;
    private final Parameter gamma;
    private final Parameter beta;

    public LayerNorm1D(int dim) {
        this(dim, ParameterFactory.float32());
    }

    public LayerNorm1D(int dim, ParameterFactory parameters) {
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        this.dim = dim;
        gamma = parameters.ones("gamma", 1, dim);
        beta = parameters.zeros("beta", 1, dim);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor matrix = PositionNormalization.matrix(input, dim);
        return PositionNormalization.restore(input, NormalizationOps.layerNorm(matrix, gamma.value, beta.value, EPS));
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(gamma, beta);
    }
}
