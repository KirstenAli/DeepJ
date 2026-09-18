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
        validateInputs(model, lossFn, opt, xAll, yAll);
        Random rnd = new Random(seed);
        return new Trainer(batchSize -> trainBatch(model, lossFn, opt, xAll, yAll, rnd, batchSize));
    }

    private static void validateInputs(Layer model, LossFunction lossFn, ParameterOptimizer opt,
                                       Tensor xAll, Tensor yAll) {
        if (model == null) throw new IllegalArgumentException("model must not be null");
        if (lossFn == null) throw new IllegalArgumentException("lossFn must not be null");
        if (opt == null) throw new IllegalArgumentException("opt must not be null");
        if (xAll == null || yAll == null) throw new IllegalArgumentException("xAll/yAll must not be null");
        if (xAll.rows != yAll.rows) throw new IllegalArgumentException("xAll.rows must equal yAll.rows");
    }

    private static float trainBatch(Layer model, LossFunction lossFn, ParameterOptimizer opt,
                                    Tensor xAll, Tensor yAll, Random rnd, int batchSize) {
        model.zeroGrad();
        TensorBatch batch = selectBatch(xAll, yAll, rnd, batchSize);
        Tensor prediction = model.forward(batch.x());
        float loss = lossFn.loss(prediction, batch.y());
        model.backward(lossFn.gradient(prediction, batch.y()));
        opt.step(model.parameters());
        return loss;
    }

    private static TensorBatch selectBatch(Tensor xAll, Tensor yAll, Random rnd, int batchSize) {
        if (batchSize >= xAll.rows) return new TensorBatch(xAll, yAll);
        int[] rows = sampleIndices(xAll.rows, batchSize, rnd);
        Tensor x = Tensor.sliceRows(xAll, rows, xAll.cols);
        Tensor y = Tensor.sliceRows(yAll, rows, yAll.cols);
        return new TensorBatch(x, y);
    }

    private static int[] sampleIndices(int rowCount, int batchSize, Random rnd) {
        int[] indices = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            indices[i] = rnd.nextInt(rowCount);
        }
        return indices;
    }

    private record TensorBatch(Tensor x, Tensor y) {}
}
