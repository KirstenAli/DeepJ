package org.deepj.ann.training;

import org.deepj.ann.Tensor;
import org.deepj.ann.layers.Layer;
import org.deepj.ann.loss.LossFunction;
import org.deepj.ann.optimisers.ParameterOptimizer;

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
                xb = sampleRows(xAll, batchSize, rnd);
                yb = sampleRows(yAll, batchSize, rnd);
            }

            Tensor pred = model.forward(xb);
            double loss = lossFn.loss(pred, yb);

            Tensor dPred = lossFn.gradient(pred, yb);
            model.backward(dPred, 0.0);

            // One optimizer step per batch
            opt.step(model.parameters());

            return loss;
        });
    }

    private static Tensor sampleRows(Tensor t, int batchSize, Random rnd) {
        Tensor out = new Tensor(batchSize, t.cols);
        for (int i = 0; i < batchSize; i++) {
            int r = rnd.nextInt(t.rows);
            if (t.cols >= 0) System.arraycopy(t.data[r], 0, out.data[i], 0, t.cols);
        }
        return out;
    }
}