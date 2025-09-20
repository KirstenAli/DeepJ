package org.DeepJ.ann.optimisers;

import org.DeepJ.ann.Tensor;

public interface Optimizer {
    Tensor apply(Tensor param, Tensor grad);
}

