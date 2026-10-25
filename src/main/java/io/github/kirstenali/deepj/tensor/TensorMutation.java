package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.Autograd;

final class TensorMutation {
    private TensorMutation() {}

    static Tensor requireGrad(Tensor tensor, boolean required) {
        tensor.gradientState().require(required);
        return tensor;
    }

    static void adamW(Tensor w, Tensor g, Tensor mt, Tensor vt, float lr, float beta1,
                       float beta2, float eps, float weightDecay, float bc1, float bc2) {
        w.gradientState().beforeMutation();
        mt.gradientState().beforeMutation();
        vt.gradientState().beforeMutation();
        Tensor.backend().adamWUpdate(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
    }

    static Tensor apply(Tensor target, Runnable operation, Tensor... operands) {
        if (Autograd.tracks(operands)) {
            throw new IllegalStateException("In-place operations cannot consume tracked operands; use noGrad");
        }
        target.gradientState().beforeMutation();
        operation.run();
        if (target.getGpuTag() instanceof GpuBuffer buffer && !buffer.cpuStale) buffer.needsUpload = true;
        return target;
    }
}
