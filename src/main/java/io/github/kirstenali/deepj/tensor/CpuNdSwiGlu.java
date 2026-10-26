package io.github.kirstenali.deepj.tensor;

/** Equal-shaped SwiGLU operands decode to FLOAT32 and only round the completed output. */
public final class CpuNdSwiGlu {
    private CpuNdSwiGlu() {}

    public static Tensor apply(Tensor gate, Tensor up) {
        Tensor.requireSameShape(gate, up, "SwiGLU");
        gate.materialize();
        up.materialize();
        Tensor output = new Tensor(DType.promote(gate.dtype(), up.dtype()), gate.shape());
        for (int index = 0; index < output.numel(); index++) {
            float input = value(gate, index);
            output.memory().set(index, input * sigmoid(input) * value(up, index));
        }
        return output;
    }

    public static SwiGluBackwardResult backward(Tensor gradient, Tensor gate, Tensor up) {
        Tensor.requireSameShape(gate, up, "SwiGLU");
        Tensor.requireSameShape(gate, gradient, "SwiGLU gradient");
        gate.materialize();
        up.materialize();
        gradient.materialize();
        Tensor gateGradient = new Tensor(DType.FLOAT32, gate.shape());
        Tensor upGradient = new Tensor(DType.FLOAT32, up.shape());
        for (int index = 0; index < gradient.numel(); index++) {
            float input = value(gate, index), sigmoid = sigmoid(input), activated = input * sigmoid;
            float derivative = sigmoid + activated * (1.0f - sigmoid), upstream = value(gradient, index);
            gateGradient.memory().set(index, (upstream * value(up, index)) * derivative);
            upGradient.memory().set(index, upstream * activated);
        }
        return new SwiGluBackwardResult(gateGradient, upGradient);
    }

    private static float sigmoid(float value) { return 1.0f / (1.0f + (float) Math.exp(-value)); }
    private static float value(Tensor tensor, int index) {
        return tensor.memory().get(tensor.layout().addressAt(index));
    }
}
