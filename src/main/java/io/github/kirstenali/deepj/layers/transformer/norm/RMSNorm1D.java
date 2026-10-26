package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.autograd.NormalizationOps;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;

public final class RMSNorm1D implements NormLayer {
    private static final float EPS = 1e-6f;
    private final int dim;
    private final Parameter gamma;

    public RMSNorm1D(int dim) {
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        this.dim = dim;
        gamma = new Parameter(Tensor.ones(1, dim));
    }

    @Override
    public Tensor forward(Tensor input) {
        if (input.matrixCols() != dim) {
            throw new IllegalArgumentException("Expected cols=" + dim + " got " + input.matrixCols());
        }
        return NormalizationOps.rmsNorm(input, gamma.value, EPS);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(gamma);
    }
}
