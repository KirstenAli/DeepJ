package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.models.CausalLM;
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

    public static Trainer trainer(CausalLM model, TextDataset dataset, float lr) {
        ParameterOptimizer opt = AdamW.defaultAdamW(lr);

        return new Trainer(batchSize -> {
            model.zeroGrad();
            Batch batch = dataset.nextBatch(batchSize);
            List<Parameter> params = model.parameters();
            float lossSum = accumulateBatchLossAndBackward(model, batch, batchSize);
            float avgLoss = computeAverageLoss(lossSum, batchSize);
            validateFiniteLoss(avgLoss);
            averageGradients(params, batchSize);
            clipGradientsGlobally(params, model.gradClipNorm());

            opt.step(params);

            return avgLoss;
        });
    }

    private static float accumulateBatchLossAndBackward(CausalLM model, Batch batch, int batchSize) {
        float lossSum = 0.0f;

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

    private static float computeAverageLoss(float lossSum, int batchSize) {
        return lossSum / batchSize;
    }

    private static void validateFiniteLoss(float avgLoss) {
        if (!Float.isFinite(avgLoss)) {
            throw new IllegalStateException("Non-finite loss encountered during training");
        }
    }

    private static void averageGradients(List<Parameter> params, int batchSize) {
        scaleGradients(params, 1.0f / batchSize);
    }

    private static void clipGradientsGlobally(List<Parameter> params, float maxNorm) {
        float gradNorm = computeGlobalGradNorm(params);
        if (gradNorm > maxNorm) {
            float scale = maxNorm / (gradNorm + 1e-12f);
            scaleGradients(params, scale);
        }
    }

    private static float computeGlobalGradNorm(List<Parameter> params) {
        float gradNormSq = 0.0f;
        for (Parameter p : params) {
            if (p.grad != null) {
                p.grad.materialize();
                float l2sq = 0.0f;
                for (float v : p.grad.data) {
                    l2sq += v * v;
                }
                if (!Float.isFinite(l2sq)) {
                    throw new IllegalStateException("Non-finite gradient encountered during training");
                }
                gradNormSq += l2sq;
            }
        }
        return fSqrt(gradNormSq);
    }

    private static float fSqrt(float x) { return (float) Math.sqrt(x); }

    private static void scaleGradients(List<Parameter> params, float scale) {
        for (Parameter p : params) {
            if (p.grad != null) {
                p.grad.multiplyScalarInPlace(scale);
            }
        }
    }
}