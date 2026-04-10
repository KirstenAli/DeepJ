package io.github.kirstenali.deepj.optimisers;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * AdamW optimizer with per-parameter state.
 *
 * <p>State is keyed by Parameter identity, and updates are applied in-place.
 */
public final class AdamW implements ParameterOptimizer {

    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;

    private long step = 0;

    private final Map<Parameter, Tensor> m = new IdentityHashMap<>();
    private final Map<Parameter, Tensor> v = new IdentityHashMap<>();

    public AdamW(float lr, float beta1, float beta2, float eps, float weightDecay) {
        validateHyperparameters(lr, beta1, beta2, eps);

        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    public static AdamW defaultAdamW(float lr) {
        return new AdamW(lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    @Override
    public void step(List<Parameter> params) {
        if (params == null) {
            throw new IllegalArgumentException("params must not be null");
        }

        step++;

        float bc1 = 1.0f - (float) Math.pow(beta1, step);
        float bc2 = 1.0f - (float) Math.pow(beta2, step);

        for (Parameter p : params) {
            if (p != null) {
                stepParam(p, bc1, bc2);
            }
        }
    }

    private void validateHyperparameters(float lr, float beta1, float beta2, float eps) {
        if (lr <= 0) {
            throw new IllegalArgumentException("lr must be > 0");
        }
        if (beta1 <= 0 || beta1 >= 1) {
            throw new IllegalArgumentException("beta1 must be in (0,1)");
        }
        if (beta2 <= 0 || beta2 >= 1) {
            throw new IllegalArgumentException("beta2 must be in (0,1)");
        }
        if (eps <= 0) {
            throw new IllegalArgumentException("eps must be > 0");
        }
    }

    private void stepParam(Parameter p, float bc1, float bc2) {
        Tensor w = p.value;
        Tensor g = p.grad;

        if (w == null || g == null) {
            return;
        }

        validateParamShapes(w, g);

        Tensor mt = m.computeIfAbsent(p, __ -> Tensor.zeros(w.rows, w.cols));
        Tensor vt = v.computeIfAbsent(p, __ -> Tensor.zeros(w.rows, w.cols));

        Tensor.adamWUpdate(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
    }

    private void validateParamShapes(Tensor w, Tensor g) {
        if (w.rows != g.rows || w.cols != g.cols) {
            throw new IllegalArgumentException("grad shape must match param shape");
        }
    }
}