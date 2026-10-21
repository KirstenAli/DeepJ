package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

record SavedTensor(Tensor tensor, long version) {
    SavedTensor(Tensor tensor, BackwardNode node) {
        this(tensor, tensor.gradientState().version());
        tensor.gradientState().saveFor(node);
    }

    void validate() {
        if (tensor.gradientState().version() != version) {
            throw new IllegalStateException("A tensor needed for backward was modified after forward");
        }
    }
}
