package org.DeepJ.ann.optimisers;

import org.DeepJ.ann.Tensor;

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * AdamW optimizer with per-parameter state.
 * Suitable default for transformer training.
 */
public class AdamW implements ParameterOptimizer {

    private final double lr;
    private final double beta1;
    private final double beta2;
    private final double eps;
    private final double weightDecay;

    private long step = 0;

    private final Map<Tensor, Tensor> m = new IdentityHashMap<>();
    private final Map<Tensor, Tensor> v = new IdentityHashMap<>();

    public AdamW(double lr, double beta1, double beta2, double eps, double weightDecay) {
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
    public void step(Parameter p) {
        step++;

        Tensor param = p.value;
        Tensor grad = p.grad;

        Tensor mt = m.computeIfAbsent(param, t -> Tensor.zeros(t.rows, t.cols));
        Tensor vt = v.computeIfAbsent(param, t -> Tensor.zeros(t.rows, t.cols));

        // m = beta1*m + (1-beta1)*g
        mt = mt.multiplyScalar(beta1).add(grad.multiplyScalar(1.0 - beta1));
        // v = beta2*v + (1-beta2)*g^2
        Tensor g2 = grad.multiply(grad);
        vt = vt.multiplyScalar(beta2).add(g2.multiplyScalar(1.0 - beta2));

        m.put(param, mt);
        v.put(param, vt);

        // bias correction
        double bc1 = 1.0 - Math.pow(beta1, step);
        double bc2 = 1.0 - Math.pow(beta2, step);

        Tensor mHat = mt.divideScalar(bc1);
        Tensor vHat = vt.divideScalar(bc2);

        Tensor denom = vHat.sqrt().addScalar(eps);

        // Adam step
        Tensor update = mHat.divide(denom).multiplyScalar(lr);

        // AdamW decay (decoupled)
        if (weightDecay != 0.0) {
            update = update.add(param.multiplyScalar(lr * weightDecay));
        }

        p.value = param.subtract(update);
    }
}
