package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * AdamW optimizer with per-parameter state.
 * Correct bias correction and decoupled weight decay.
 *
 * <p>State is keyed by Parameter identity, and updates are applied in-place.
 */
public final class AdamW implements ParameterOptimizer {

    private final double lr;
    private final double beta1;
    private final double beta2;
    private final double eps;
    private final double weightDecay;

    private long step = 0;

    private final Map<Parameter, Tensor> m = new IdentityHashMap<>();
    private final Map<Parameter, Tensor> v = new IdentityHashMap<>();

    public AdamW(double lr, double beta1, double beta2, double eps, double weightDecay) {
        if (lr <= 0) throw new IllegalArgumentException("lr must be > 0");
        if (beta1 <= 0 || beta1 >= 1) throw new IllegalArgumentException("beta1 must be in (0,1)");
        if (beta2 <= 0 || beta2 >= 1) throw new IllegalArgumentException("beta2 must be in (0,1)");
        if (eps <= 0) throw new IllegalArgumentException("eps must be > 0");

        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    public static AdamW defaultAdamW(double lr) {
        return new AdamW(lr, 0.9, 0.999, 1e-8, 0.01);
    }

    @Override
    public void step(List<Parameter> params) {
        if (params == null) throw new IllegalArgumentException("params must not be null");

        step++;

        // bias correction scalars for this optimizer step
        double bc1 = 1.0 - Math.pow(beta1, step);
        double bc2 = 1.0 - Math.pow(beta2, step);

        for (Parameter p : params) {
            if (p == null) continue;
            stepParam(p, bc1, bc2);
        }
    }

    private void stepParam(Parameter p, double bc1, double bc2) {
        Tensor w = p.value;
        Tensor g = p.grad;

        if (w == null || g == null) return;
        if (w.rows != g.rows || w.cols != g.cols) {
            throw new IllegalArgumentException("grad shape must match param shape");
        }

        Tensor mt = m.computeIfAbsent(p, __ -> Tensor.zeros(w.rows, w.cols));
        Tensor vt = v.computeIfAbsent(p, __ -> Tensor.zeros(w.rows, w.cols));

        for (int r = 0; r < w.rows; r++) {
            for (int c = 0; c < w.cols; c++) {
                double grad = g.data[r][c];

                // Update biased moments
                double mNew = beta1 * mt.data[r][c] + (1.0 - beta1) * grad;
                double vNew = beta2 * vt.data[r][c] + (1.0 - beta2) * (grad * grad);

                mt.data[r][c] = mNew;
                vt.data[r][c] = vNew;

                // Bias-corrected moments
                double mHat = mNew / bc1;
                double vHat = vNew / bc2;

                // Adam step
                double update = (lr * mHat) / (Math.sqrt(vHat) + eps);

                // Decoupled weight decay (AdamW)
                if (weightDecay != 0.0) {
                    update += lr * weightDecay * w.data[r][c];
                }

                // In-place update
                w.data[r][c] -= update;
            }
        }
    }
}