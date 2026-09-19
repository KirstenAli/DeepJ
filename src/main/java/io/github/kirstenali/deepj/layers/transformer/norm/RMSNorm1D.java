package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.RmsNormResult;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;

public final class RMSNorm1D implements NormLayer {

    private static final float EPS = 1e-6f;

    private final int dim;
    private final Parameter gamma;

    private Tensor xHat;
    private Tensor rms;

    public RMSNorm1D(int dim) {
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        this.dim = dim;
        this.gamma = new Parameter(Tensor.ones(1, dim));
    }

    @Override
    public Tensor forward(Tensor x) {
        if (x.cols != dim) {
            throw new IllegalArgumentException("Expected cols=" + dim + " got " + x.cols);
        }

        RmsNormResult result = Tensor.backend().rmsNorm(x, gamma.value, EPS);
        rms = result.rms();
        xHat = result.normalized();
        return result.output();
    }

    @Override
    public Tensor backward(Tensor gradOut) {

        gamma.grad.addInPlace(gradOut.multiply(xHat).sumRows());

        return Tensor.backend().rmsNormBackward(gradOut, xHat, rms, gamma.value);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(gamma);
    }
}
