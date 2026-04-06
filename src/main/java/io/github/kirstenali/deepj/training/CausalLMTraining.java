package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;

import java.util.List;

/**
 * Wiring helpers for causal language model training.
 */
public final class CausalLMTraining {

    private CausalLMTraining() {}

    public static Trainer trainer(GPTModel model, TextDataset dataset, double lr) {
        ParameterOptimizer opt = AdamW.defaultAdamW(lr);

        return new Trainer(batchSize -> {
            model.zeroGrad();
            Batch batch = dataset.nextBatch(batchSize);
            List<Parameter> params = model.parameters();
            double lossSum = accumulateBatchLossAndBackward(model, batch, batchSize);
            double avgLoss = computeAverageLoss(lossSum, batchSize);
            validateFiniteLoss(avgLoss);
            averageGradients(params, batchSize);
            clipGradientsGlobally(params, model.gradClipNorm());

            // One optimizer step per batch
            opt.step(params);

            return avgLoss;
        });
    }

    private static double accumulateBatchLossAndBackward(GPTModel model, Batch batch, int batchSize) {
        double lossSum = 0.0;

        for (int b = 0; b < batchSize; b++) {
            int[] x = batch.x()[b];
            int[] y = batch.y()[b];

            Tensor logits = model.forward(x);
            lossSum += CrossEntropyLoss.loss(logits, y);

            Tensor dLogits = CrossEntropyLoss.gradient(logits, y);
            model.backward(dLogits);
        }

        return lossSum;
    }

    private static double computeAverageLoss(double lossSum, int batchSize) {
        return lossSum / batchSize;
    }

    private static void validateFiniteLoss(double avgLoss) {
        if (!Double.isFinite(avgLoss)) {
            throw new IllegalStateException("Non-finite loss encountered during training");
        }
    }

    private static void averageGradients(List<Parameter> params, int batchSize) {
        scaleGradients(params, 1.0 / batchSize);
    }

    private static void clipGradientsGlobally(List<Parameter> params, double maxNorm) {
        double gradNorm = computeGlobalGradNorm(params);
        if (gradNorm > maxNorm) {
            double scale = maxNorm / (gradNorm + 1e-12);
            scaleGradients(params, scale);
        }
    }

    private static double computeGlobalGradNorm(List<Parameter> params) {
        double gradNormSq = 0.0;
        for (Parameter p : params) {
            if (p.grad != null) {
                double l2sq = p.grad.multiply(p.grad).sum();
                if (!Double.isFinite(l2sq)) {
                    throw new IllegalStateException("Non-finite gradient encountered during training");
                }
                gradNormSq += l2sq;
            }
        }
        return Math.sqrt(gradNormSq);
    }

    private static void scaleGradients(List<Parameter> params, double scale) {
        for (Parameter p : params) {
            if (p.grad != null) {
                p.grad.multiplyScalarInPlace(scale);
            }
        }
    }
}