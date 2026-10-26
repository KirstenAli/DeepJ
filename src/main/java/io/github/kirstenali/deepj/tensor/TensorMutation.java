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
        OptimizerChecks.requireState(w, g, mt, vt);
        w.gradientState().beforeMutation();
        mt.gradientState().beforeMutation();
        vt.gradientState().beforeMutation();
        Tensor.backend().adamWUpdate(w, g, mt, vt, lr, beta1, beta2, eps, weightDecay, bc1, bc2);
        markHostMutation(w);
        markHostMutation(mt);
        markHostMutation(vt);
    }

    static Tensor apply(Tensor target, Runnable operation, Tensor... operands) {
        target.layout().requireWritable();
        if (Autograd.tracks(operands)) {
            throw new IllegalStateException("In-place operations cannot consume tracked operands; use noGrad");
        }
        target.gradientState().beforeMutation();
        operation.run();
        markHostMutation(target);
        return target;
    }

    static Tensor binary(Tensor target, Tensor operand, Runnable operation) {
        if (!target.hasSameShape(operand)) {
            TensorShape shape = ShapeTransforms.broadcast(target.layout().shape(), operand.layout().shape());
            if (!shape.equals(target.layout().shape())) {
                throw new IllegalArgumentException("Cannot expand in-place output");
            }
        }
        if (DType.promote(target.dtype(), operand.dtype()) != target.dtype()) {
            throw new IllegalArgumentException("In-place output cannot narrow the promoted dtype");
        }
        return apply(target, operation, operand);
    }

    private static void markHostMutation(Tensor target) {
        if (target.getGpuTag() instanceof GpuBuffer buffer && !buffer.cpuStale) buffer.needsUpload = true;
    }

    static Tensor indexAdd(Tensor target, int axis, int[] indices, Tensor updates) {
        int[] saved = indices.clone();
        IndexSelection.requireUpdates(target, updates, axis, saved);
        return apply(target, () -> Tensor.backend().indexAdd(target, axis, saved, updates), updates);
    }

    static Tensor assign(Tensor target, Tensor source) {
        Tensor.requireSameShape(target, source, "Assignment");
        return apply(target, () -> Tensor.backend().copyInto(target, source), source);
    }
}
