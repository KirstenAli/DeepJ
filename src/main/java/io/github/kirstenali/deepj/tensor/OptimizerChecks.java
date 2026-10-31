package io.github.kirstenali.deepj.tensor;

/** Adam updates accumulate in independent FLOAT32 parameter/master and moment buffers. */
public final class OptimizerChecks {
    private OptimizerChecks() {}

    public static void requireState(Tensor weight, Tensor gradient, Tensor first, Tensor second) {
        requireFloat32(weight);
        requireBuffer(weight, gradient);
        requireBuffer(weight, first);
        requireBuffer(weight, second);
        weight.layout().requireWritable();
        first.layout().requireWritable();
        second.layout().requireWritable();
        independent(weight, gradient, first, second);
    }

    private static void independent(Tensor w, Tensor g, Tensor m, Tensor v) {
        if (w.memory() == g.memory() || w.memory() == m.memory() || w.memory() == v.memory()
                || g.memory() == m.memory() || g.memory() == v.memory() || m.memory() == v.memory()) {
            throw new IllegalArgumentException("AdamW weights, gradients and moments cannot share storage");
        }
    }

    private static void requireBuffer(Tensor weight, Tensor tensor) {
        requireFloat32(tensor);
        Tensor.requireSameShape(weight, tensor, "AdamW state");
    }

    private static void requireFloat32(Tensor tensor) {
        if (tensor.dtype() != DType.FLOAT32) throw new IllegalArgumentException("AdamW update buffers must be FLOAT32");
    }
}
