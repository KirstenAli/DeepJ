package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.Autograd;

final class TensorMutation {
    private TensorMutation() {}

    static Tensor apply(Tensor target, Runnable operation, Tensor... operands) {
        if (Autograd.tracks(operands)) {
            throw new IllegalStateException("In-place operations cannot consume tracked operands; use noGrad");
        }
        target.gradientState().beforeMutation();
        operation.run();
        return target;
    }
}
