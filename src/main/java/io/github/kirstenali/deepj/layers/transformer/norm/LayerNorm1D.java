package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;

/**
 * LayerNorm over feature dimension (cols) with trainable gamma/beta exposed as {@link Parameter}s.
 */
public final class LayerNorm1D implements Layer {

    private static final double EPS = 1e-5;

    private final int dim;

    private final Parameter gamma; // 1 x dim
    private final Parameter beta;  // 1 x dim

    private Tensor x;
    private Tensor mean;
    private Tensor var;
    private Tensor xHat;
    private Tensor std;

    public LayerNorm1D(int dim) {
        this.dim = dim;
        this.gamma = new Parameter(Tensor.ones(1, dim));
        this.beta = new Parameter(Tensor.zeros(1, dim));
    }

    @Override
    public Tensor forward(Tensor x) {
        validateInput(x);
        this.x = x;

        computeStatistics(x);
        xHat = normalize(x);

        return applyAffine(xHat);
    }

    @Override
    public Tensor backward(Tensor gradOut) {
        accumulateParameterGrads(gradOut);
        Tensor dXHat = gradOut.multiplyBroadcastRows(gamma.value);
        return Tensor.layerNormBackward(dXHat, xHat, std, dim);
    }

    private void validateInput(Tensor x) {
        if (x.cols != dim) {
            throw new IllegalArgumentException("Expected cols=" + dim + " got " + x.cols);
        }
    }

    private void computeStatistics(Tensor x) {
        mean = x.meanAlongRows();
        var = x.varianceAlongRows();
        std = var.addScalar(EPS).sqrt();
    }

    private Tensor normalize(Tensor x) {
        return x.subtractBroadcastCols(mean).divideBroadcastCols(std);
    }

    private Tensor applyAffine(Tensor normalized) {
        return normalized
                .multiplyBroadcastRows(gamma.value)
                .addBroadcastRows(beta.value);
    }

    private void accumulateParameterGrads(Tensor gradOut) {
        gamma.grad = gamma.grad.add(xHat.multiply(gradOut).sumRows());
        beta.grad = beta.grad.add(gradOut.sumRows());
    }


    @Override
    public List<Parameter> parameters() {
        return List.of(gamma, beta);
    }
}