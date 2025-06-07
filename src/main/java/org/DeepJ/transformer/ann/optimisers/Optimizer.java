package org.DeepJ.transformer.ann.optimisers;

import org.DeepJ.transformer.Tensor;

public interface Optimizer {
    Tensor apply(Tensor param, Tensor grad);
}

