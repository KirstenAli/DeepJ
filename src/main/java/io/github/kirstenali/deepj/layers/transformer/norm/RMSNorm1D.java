package io.github.kirstenali.deepj.layers.transformer.norm;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.List;

/**
 * Root-Mean-Square Layer Normalisation — the normalisation used in Llama, Mistral, Qwen, and DeepSeek.
 *
 * <p>Differs from {@link LayerNorm1D} only in that it omits the mean-subtraction step,
 * normalising purely by RMS and scaling by a learnable {@code gamma} (no {@code beta}).
 *
 * <p>Forward:
 * <pre>
 *   rms  = sqrt( mean(x²) + ε )         shape: [seqLen × 1]
 *   x̂   = x / rms                        shape: [seqLen × dim]
 *   out  = γ · x̂                          shape: [seqLen × dim]
 * </pre>
 *
 * <p>Backward derivation (per row, feature index k):
 * <pre>
 *   g    = γ · dOut             (upstream scaled by gamma)
 *   dL/dx = ( g − x̂ · mean(g · x̂) ) / rms
 * </pre>
 */
public final class RMSNorm1D implements Layer {

    private static final double EPS = 1e-6;

    private final int dim;
    private final Parameter gamma;  // [1 × dim], initialised to ones; no beta

    // Forward cache (needed for backward)
    private Tensor xHat;   // [seqLen × dim]
    private Tensor rms;    // [seqLen × 1]

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

        // mean(x²) per row → [seqLen × 1]
        Tensor meanSq = x.multiply(x).meanAlongRows();
        this.rms  = meanSq.addScalar(EPS).sqrt();      // [seqLen × 1]
        this.xHat = x.divideBroadcastCols(rms);        // [seqLen × dim]

        return xHat.multiplyBroadcastRows(gamma.value);
    }

    @Override
    public Tensor backward(Tensor gradOut) {
        // Accumulate gamma gradient: sum over rows of (gradOut · x̂)
        gamma.grad = gamma.grad.add(gradOut.multiply(xHat).sumRows());

        // Scale upstream gradient by gamma: g = γ · gradOut  [seqLen × dim]
        Tensor g = gradOut.multiplyBroadcastRows(gamma.value);

        // inner product per row: mean(g · x̂)  →  [seqLen × 1]
        Tensor innerProd = g.multiply(xHat).meanAlongRows();

        // dL/dx = ( g − x̂ · innerProd ) / rms
        return g.subtract(xHat.multiplyBroadcastCols(innerProd))
                .divideBroadcastCols(rms);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(gamma);  // no beta — intentional
    }
}

