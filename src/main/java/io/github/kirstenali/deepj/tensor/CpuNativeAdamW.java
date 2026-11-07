package io.github.kirstenali.deepj.tensor;

/** Independent FLOAT32 lanes retain the Java optimizer's operation order without fused multiply-add. */
public final class CpuNativeAdamW {
    private CpuNativeAdamW() {}

    public static boolean update(Tensor weight, Tensor gradient, Tensor first, Tensor second, float learningRate,
                                 float beta1, float beta2, float epsilon, float decay, float bc1, float bc2) {
        if (!CpuBlas.AVAILABLE || weight.numel() < 256 || !dense(weight, gradient) || !dense(first, second)) {
            return false;
        }
        OptimizerChecks.requireState(weight, gradient, first, second);
        weight.materialize();
        gradient.materialize();
        first.materialize();
        second.materialize();
        return CpuBlas.adamW(weight.memory().floats(), weight.storageOffset(),
                gradient.memory().floats(), gradient.storageOffset(), first.memory().floats(), first.storageOffset(),
                second.memory().floats(), second.storageOffset(), weight.numel(),
                learningRate, beta1, beta2, epsilon, decay, bc1, bc2);
    }

    private static boolean dense(Tensor left, Tensor right) {
        return left.dtype() == DType.FLOAT32 && right.dtype() == DType.FLOAT32
                && left.isContiguous() && right.isContiguous();
    }
}
