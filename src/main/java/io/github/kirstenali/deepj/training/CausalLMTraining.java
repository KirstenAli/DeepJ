package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.gpt.Batch;
import io.github.kirstenali.deepj.gpt.GPTModel;
import io.github.kirstenali.deepj.gpt.TextDataset;
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