package org.deepj.ann.training;

import org.deepj.ann.Tensor;
import org.deepj.ann.gpt.Batch;
import org.deepj.ann.gpt.GPTModel;
import org.deepj.ann.gpt.TextDataset;
import org.deepj.ann.loss.CrossEntropyLoss;
import org.deepj.ann.optimisers.AdamW;
import org.deepj.ann.optimisers.Parameter;
import org.deepj.ann.optimisers.ParameterOptimizer;

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

            double lossSum = 0.0;

            // Accumulate grads over batch
            for (int b = 0; b < batchSize; b++) {
                int[] x = batch.x()[b];
                int[] y = batch.y()[b];

                Tensor logits = model.forward(x);
                lossSum += CrossEntropyLoss.loss(logits, y);

                Tensor dLogits = CrossEntropyLoss.gradient(logits, y);
                model.backward(dLogits);
            }

            // Average gradients
            List<Parameter> params = model.parameters();
            for (Parameter p : params) {
                if (p.grad != null) {
                    p.grad = p.grad.divideScalar(batchSize);
                }
            }

            // One optimizer step per batch
            opt.step(params);

            return lossSum / batchSize;
        });
    }
}