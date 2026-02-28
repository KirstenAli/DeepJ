package org.DeepJ.ann.optimisers;

import org.DeepJ.ann.Tensor;

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * SGD with momentum. Keeps a separate velocity tensor per parameter.
 */
public class SGDMomentum implements Optimizer, ParameterOptimizer {

    private final double lr;
    private final double momentum;

    private final Map<Tensor, Tensor> velocities = new IdentityHashMap<>();

    public SGDMomentum(double lr, double momentum) {
        this.lr = lr;
        this.momentum = momentum;
    }

    @Override
    public Tensor apply(Tensor param, Tensor grad) {
        Tensor v = velocities.computeIfAbsent(param, p -> Tensor.zeros(p.rows, p.cols));
        v = v.multiplyScalar(momentum).subtract(grad.multiplyScalar(lr));
        velocities.put(param, v);
        return param.add(v);
    }

    @Override
    public void step(Parameter parameter) {
        parameter.value = apply(parameter.value, parameter.grad);
    }
}
