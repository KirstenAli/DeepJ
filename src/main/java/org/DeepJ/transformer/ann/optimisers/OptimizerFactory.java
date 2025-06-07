package org.DeepJ.transformer.ann.optimisers;

public interface OptimizerFactory {
    Optimizer create(int rows, int cols);
}