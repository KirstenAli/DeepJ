package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.BatchSource;
import io.github.kirstenali.deepj.nn.transformer.CausalLM;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;

/** Runs several micro-batches per optimizer step so the effective batch is microBatch × steps. */
public final class GradientAccumulation {
    private GradientAccumulation() {}

    public static Trainer trainer(CausalLM model, BatchSource dataset, float lr, int steps) {
        return trainer(model, dataset, AdamW.defaultAdamW(lr), steps);
    }

    public static Trainer trainer(CausalLM model, BatchSource dataset, ParameterOptimizer optimizer, int steps) {
        if (model == null || dataset == null || optimizer == null) {
            throw new IllegalArgumentException("model, dataset, and optimizer must not be null");
        }
        if (steps <= 0) throw new IllegalArgumentException("accumulation steps must be > 0");
        return new Trainer(microBatch ->
                CausalLMTraining.accumulatedStep(model, dataset, optimizer, microBatch, steps));
    }
}

