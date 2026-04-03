package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.loss.LossFunction;
import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;

import java.util.Random;

/**
 * Helpers to train classic Tensor->Tensor supervised models (e.g., FNN)
 * using the unified Trainer wrapper.
 */
public final class SupervisedTraining {

    private SupervisedTraining() {}

    public static Trainer trainer(
            Layer model,
            LossFunction lossFn,
            ParameterOptimizer opt,
            Tensor xAll,
            Tensor yAll,
            long seed
    ) {
        if (model == null) throw new IllegalArgumentException("model must not be null");
        if (lossFn == null) throw new IllegalArgumentException("lossFn must not be null");
        if (opt == null) throw new IllegalArgumentException("opt must not be null");
        if (xAll == null || yAll == null) throw new IllegalArgumentException("xAll/yAll must not be null");
        if (xAll.rows != yAll.rows) throw new IllegalArgumentException("xAll.rows must equal yAll.rows");

        Random rnd = new Random(seed);

        return new Trainer(batchSize -> {
            model.zeroGrad();

            Tensor xb, yb;
            if (batchSize >= xAll.rows) {
                xb = xAll;
                yb = yAll;
            } else {
                xb = Tensor.sampleRows(xAll, batchSize, rnd);
                yb = Tensor.sampleRows(yAll, batchSize, rnd);
            }

            Tensor pred = model.forward(xb);
            double loss = lossFn.loss(pred, yb);

            Tensor dPred = lossFn.gradient(pred, yb);
            model.backward(dPred);

            // One optimizer step per batch
            opt.step(model.parameters());

            return loss;
        });
    }
}