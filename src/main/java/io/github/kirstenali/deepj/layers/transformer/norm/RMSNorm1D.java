package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.layers.transformer.norm.NormLayer;
import io.github.kirstenali.deepj.optimisers.Parameter;
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

        Tensor meanSq = x.multiply(x).meanAlongRows();
        this.rms  = meanSq.addScalar(EPS).sqrt();
        this.xHat = x.divideBroadcastCols(rms);

        return xHat.multiplyBroadcastRows(gamma.value);
    }

    @Override
    public Tensor backward(Tensor gradOut) {

        gamma.grad.addInPlace(gradOut.multiply(xHat).sumRows());

        Tensor g = gradOut.multiplyBroadcastRows(gamma.value);

        Tensor innerProd = g.multiply(xHat).meanAlongRows();

        return g.subtract(xHat.multiplyBroadcastCols(innerProd))
                .divideBroadcastCols(rms);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(gamma);
    }
}
