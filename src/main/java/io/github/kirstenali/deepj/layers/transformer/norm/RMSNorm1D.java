package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.autograd.NormalizationOps;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;

public final class RMSNorm1D implements NormLayer {
    private static final float EPS = 1e-6f;
    private final int dim;
    private final Parameter gamma;

    public RMSNorm1D(int dim) {
        this(dim, ParameterFactory.float32());
    }

    public RMSNorm1D(int dim, ParameterFactory parameters) {
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        this.dim = dim;
        gamma = parameters.ones("gamma", 1, dim);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor matrix = PositionNormalization.matrix(input, dim);
        return PositionNormalization.restore(input, NormalizationOps.rmsNorm(matrix, gamma.value, EPS));
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(gamma);
    }
}
