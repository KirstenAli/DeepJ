package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.CrossEntropyResult;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.BatchSource;
import io.github.kirstenali.deepj.models.CausalLM;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;

import java.util.List;

public final class CausalLMTraining {

    private CausalLMTraining() {}

    public static Trainer trainer(CausalLM model, BatchSource dataset, float lr) {
        return trainer(model, dataset, AdamW.defaultAdamW(lr));
    }

    public static Trainer trainer(CausalLM model, BatchSource dataset, float lr,
                                  int accumulationSteps) {
        return trainer(model, dataset, AdamW.defaultAdamW(lr), accumulationSteps);
    }

    public static Trainer trainer(CausalLM model, BatchSource dataset, ParameterOptimizer optimizer) {
        return trainer(model, dataset, optimizer, 1);
    }

    public static Trainer trainer(CausalLM model, BatchSource dataset,
                                  ParameterOptimizer optimizer, int accumulationSteps) {
        validateTrainerArgs(model, dataset, optimizer, accumulationSteps);
        return new Trainer(batchSize -> trainStep(model, dataset, optimizer,
                effectiveBatchSize(batchSize, accumulationSteps)));
    }

    private static void validateTrainerArgs(CausalLM model, BatchSource dataset,
                                            ParameterOptimizer optimizer, int accumulationSteps) {
        if (model == null || dataset == null || optimizer == null) {
            throw new IllegalArgumentException("model, dataset, and optimizer must not be null");
        }
        if (accumulationSteps < 1) {
            throw new IllegalArgumentException("accumulationSteps must be positive");
        }
    }

    private static int effectiveBatchSize(int batchSize, int accumulationSteps) {
        return Math.multiplyExact(batchSize, accumulationSteps);
    }

    private static float trainStep(CausalLM model, BatchSource dataset,
                                   ParameterOptimizer optimizer, int batchSize) {
        model.zeroGrad();
        Batch batch = dataset.nextBatch(batchSize);
        List<Parameter> params = model.parameters();
        float loss = computeAverageLoss(accumulateBatchLossAndBackward(model, batch, batchSize), batchSize);
        validateFiniteLoss(loss);
        averageGradients(params, batchSize);
        clipGradientsGlobally(params, model.gradClipNorm());
        optimizer.step(params);
        return loss;
    }

    private static float accumulateBatchLossAndBackward(CausalLM model, Batch batch, int batchSize) {
        float lossSum = 0.0f;

        for (int b = 0; b < batchSize; b++) {
            int[] x = batch.x()[b];
            int[] y = batch.y()[b];
            Tensor logits = model.forward(x);
            boolean[] mask = batch.mask(b);
            lossSum += backwardAndLoss(model, logits, y, mask);
        }
        return lossSum;
    }

    private static float backwardAndLoss(CausalLM model, Tensor logits,
                                         int[] targets, boolean[] mask) {
        if (mask != null) {
            float loss = loss(logits, targets, mask);
            model.backward(gradient(logits, targets, mask));
            return loss;
        }
        CrossEntropyResult result = CrossEntropyLoss.result(logits, targets);
        model.backward(result.gradient());
        return result.meanLoss();
    }

    private static float loss(Tensor logits, int[] targets, boolean[] mask) {
        return mask == null ? CrossEntropyLoss.loss(logits, targets)
                : CrossEntropyLoss.loss(logits, targets, mask);
    }

    private static Tensor gradient(Tensor logits, int[] targets, boolean[] mask) {
        return mask == null ? CrossEntropyLoss.gradient(logits, targets)
                : CrossEntropyLoss.gradient(logits, targets, mask);
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
        List<Tensor> gradients = params.stream().map(parameter -> parameter.grad)
                .filter(gradient -> gradient != null).toList();
        float norm = Tensor.backend().l2Norm(gradients);
        if (!Float.isFinite(norm)) {
            throw new IllegalStateException("Non-finite gradient encountered during training");
        }
        return norm;
    }

    private static void scaleGradients(List<Parameter> params, float scale) {
        for (Parameter p : params) {
            if (p.grad != null) {
                p.grad.multiplyScalarInPlace(scale);
            }
        }
    }
}
