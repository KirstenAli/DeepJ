package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

/** An optional VJP fast path that writes directly into a leaf's gradient buffer. */
interface LeafGradientRule extends BackwardRule {
    boolean accumulate(Tensor gradient);
}
