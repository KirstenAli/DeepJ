package org.DeepJ.annv2.optimisers;

import org.DeepJ.transformer.Tensor;

public interface Optimizer {
    Tensor apply(Tensor param, Tensor grad);
}

