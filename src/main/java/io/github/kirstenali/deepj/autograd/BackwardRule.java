package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

/** A first-order vector-Jacobian product, in the same order as the operation's inputs. */
@FunctionalInterface
public interface BackwardRule {
    Tensor[] apply(Tensor gradient);
}
