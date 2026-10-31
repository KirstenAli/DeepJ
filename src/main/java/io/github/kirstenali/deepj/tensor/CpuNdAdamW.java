package io.github.kirstenali.deepj.tensor;

/** Logical-layout Adam arithmetic matches the original contiguous FLOAT32 update order. */
public final class CpuNdAdamW {
    private CpuNdAdamW() {}

    public static void update(Tensor weight, Tensor gradient, Tensor first, Tensor second, float learningRate,
                                float beta1, float beta2, float epsilon, float decay, float bc1, float bc2) {
        OptimizerChecks.requireState(weight, gradient, first, second);
        weight.materialize();
        gradient.materialize();
        first.materialize();
        second.materialize();
        for (int index = 0; index < weight.numel(); index++) {
            float grad = value(gradient, index), original = value(weight, index);
            float m = beta1 * value(first, index) + (1.0f - beta1) * grad;
            float v = beta2 * value(second, index) + (1.0f - beta2) * (grad * grad);
            set(first, index, m);
            set(second, index, v);
            float update = (learningRate * (m / bc1)) / ((float) Math.sqrt(v / bc2) + epsilon);
            if (decay != 0) update += learningRate * decay * original;
            set(weight, index, original - update);
        }
    }

    private static float value(Tensor tensor, int index) {
        return tensor.memory().get(tensor.layout().addressAt(index));
    }
    private static void set(Tensor tensor, int index, float value) {
        tensor.memory().set(tensor.layout().addressAt(index), value);
    }
}
