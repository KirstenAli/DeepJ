package org.DeepJ.ann.training;

import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.layers.Layer;
import org.DeepJ.ann.loss.LossFunction;
import org.DeepJ.ann.optimisers.Parameter;
import org.DeepJ.ann.optimisers.ParameterOptimizer;

import java.util.Random;

/**
 * Helpers to train classic Tensor->Tensor supervised models (e.g., {@link org.DeepJ.ann.layers.FNN})
 * using the unified {@link Trainer} wrapper.
 *
 * <p>Supports full-batch training (batchSize >= rows) or simple random row mini-batches.
 */
public final class SupervisedTraining {

    private SupervisedTraining() {}

    /**
     * Create a {@link Trainer} for supervised regression/classification where the model maps
     * input Tensor -> output Tensor.
     *
     * @param model  differentiable model (Layer)
     * @param lossFn loss function
     * @param opt    optimizer operating on {@link Parameter}s
     * @param xAll   inputs, shape [N x in]
     * @param yAll   targets, shape [N x out]
     * @param seed   RNG seed for mini-batch sampling
     */
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
            Tensor xb;
            Tensor yb;

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

            for (Parameter p : model.parameters()) {
                opt.step(p);
            }
            model.zeroGrad();

            return loss;
        });
    }

    private static Tensor sampleRows(Tensor t, int batchSize, Random rnd) {
        Tensor out = new Tensor(batchSize, t.cols);
        for (int i = 0; i < batchSize; i++) {
            int r = rnd.nextInt(t.rows);
            for (int c = 0; c < t.cols; c++) {
                out.data[i][c] = t.data[r][c];
            }
        }
        return out;
    }
}
