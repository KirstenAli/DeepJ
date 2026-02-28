package org.DeepJ.ann.training;

import org.DeepJ.ann.Tensor;
import org.DeepJ.ann.gpt.Batch;
import org.DeepJ.ann.gpt.GPTModel;
import org.DeepJ.ann.gpt.TextDataset;
import org.DeepJ.ann.loss.CrossEntropyLoss;
import org.DeepJ.ann.optimisers.AdamW;
import org.DeepJ.ann.optimisers.Parameter;
import org.DeepJ.ann.optimisers.ParameterOptimizer;

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

            for (int b = 0; b < batchSize; b++) {
                int[] x = batch.x()[b];
                int[] y = batch.y()[b];

                Tensor logits = model.forward(x);
                lossSum += CrossEntropyLoss.loss(logits, y);

                Tensor dLogits = CrossEntropyLoss.gradient(logits, y);
                model.backward(dLogits);
            }

            List<Parameter> params = model.parameters();
            for (Parameter p : params) {
                p.grad = p.grad.divideScalar(batchSize);
                opt.step(p);
            }

            return lossSum / batchSize;
        });
    }
}
