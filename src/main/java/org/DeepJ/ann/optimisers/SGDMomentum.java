package org.DeepJ.ann.optimisers;

import org.DeepJ.ann.Tensor;

public class SGDMomentum implements Optimizer {

    private final double lr;
    private final double momentum;
    private Tensor velocity;

    public SGDMomentum(double lr, double momentum) {
        this.lr = lr;
        this.momentum = momentum;
    }

    @Override
    public Tensor apply(Tensor param, Tensor grad) {

        if (velocity == null)
            velocity = Tensor.zeros(param.rows, param.cols);

        velocity = velocity.multiplyScalar(momentum)
                .subtract(grad.multiplyScalar(lr));

        return param.add(velocity);
    }
}

