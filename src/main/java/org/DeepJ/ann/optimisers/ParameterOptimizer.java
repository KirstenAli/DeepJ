package org.DeepJ.ann.optimisers;

/**
 * Optimizer operating on {@link Parameter} so it can keep per-parameter state.
 */
public interface ParameterOptimizer {
    void step(Parameter parameter);
}
