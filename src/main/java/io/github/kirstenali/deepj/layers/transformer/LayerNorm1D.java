package io.github.kirstenali.deepj.layers.transformer;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;

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
        this.beta  = new Parameter(Tensor.zeros(1, dim));
    }

    public Tensor forward(Tensor x) {
        if (x.cols != dim) throw new IllegalArgumentException("Expected cols=" + dim + " got " + x.cols);
        this.x = x;

        mean = x.meanAlongRows();
        var  = x.varianceAlongRows();
        std  = var.addScalar(EPS).sqrt();

        xHat = x.subtractBroadcastCols(mean).divideBroadcastCols(std);
        return xHat.multiplyBroadcastRows(gamma.value).addBroadcastRows(beta.value);
    }

    public Tensor backward(Tensor gradOut) {
        // grads for gamma/beta
        gamma.grad = gamma.grad.add(xHat.multiply(gradOut).sumRows());
        beta.grad  = beta.grad.add(gradOut.sumRows());

        Tensor dXHat = gradOut.multiplyBroadcastRows(gamma.value);

        // Backprop through normalization (per row)
        int n = dim;
        Tensor dX = new Tensor(x.rows, x.cols);

        for (int r = 0; r < x.rows; r++) {
            double meanR = mean.data[r][0];
            double stdR = std.data[r][0];

            // Compute sums for efficient formula
            double sumD = 0.0;
            double sumDXHatXHat = 0.0;
            for (int c = 0; c < n; c++) {
                double d = dXHat.data[r][c];
                sumD += d;
                sumDXHatXHat += d * xHat.data[r][c];
            }

            for (int c = 0; c < n; c++) {
                double d = dXHat.data[r][c];
                double xh = xHat.data[r][c];
                // (1/std) * (d - mean(d) - xhat * mean(d*xhat))
                dX.data[r][c] = (1.0 / stdR) * (d - sumD / n - xh * (sumDXHatXHat / n));
            }
        }

        return dX;
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(gamma, beta);
    }

    @Override
    public Tensor backward(Tensor gradOutput, double learningRate) {
        return backward(gradOutput);
    }

}
