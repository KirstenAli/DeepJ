package org.DeepJ.ann.optimisers;

public interface OptimizerFactory {
    Optimizer create(int rows, int cols);
}