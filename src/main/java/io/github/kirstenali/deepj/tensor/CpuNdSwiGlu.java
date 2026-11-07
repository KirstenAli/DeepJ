package io.github.kirstenali.deepj.tensor;

/** Equal-shaped SwiGLU operands decode to FLOAT32 and only round the completed output. */
public final class CpuNdSwiGlu {
    private CpuNdSwiGlu() {}

    public static Tensor apply(Tensor gate, Tensor up) {
        Tensor.requireSameShape(gate, up, "SwiGLU");
        materialize(gate, up);
        Tensor output = new Tensor(DType.promote(gate.dtype(), up.dtype()), gate.shape());
        if (CpuCompactPointwise.apply(gate, up, output, CpuCompactPointwise.SWIGLU)) return output;
        if (CpuDenseElementwise.supports(gate, up)) {
            CpuDenseSwiGlu.forward(gate, up, output);
            return output;
        }
        for (int index = 0; index < output.numel(); index++) {
            float input = value(gate, index);
            output.memory().set(index, input * sigmoid(input) * value(up, index));
        }
        return output;
    }

    public static SwiGluBackwardResult backward(Tensor gradient, Tensor gate, Tensor up) {
        Tensor.requireSameShape(gate, up, "SwiGLU");
        Tensor.requireSameShape(gate, gradient, "SwiGLU gradient");
        materialize(gate, up, gradient);
        var result = new SwiGluBackwardResult(new Tensor(DType.FLOAT32, gate.shape()),
                new Tensor(DType.FLOAT32, up.shape()));
        if (CpuCompactSwiGluBackward.apply(gradient, gate, up, result)) return result;
        if (CpuDenseElementwise.supports(gradient, gate, up)) {
            CpuDenseSwiGlu.backward(gradient, gate, up, result);
        } else {
            backwardRange(gradient, gate, up, result, 0, gradient.numel());
        }
        return result;
    }

    static void backwardRange(Tensor gradient, Tensor gate, Tensor up, SwiGluBackwardResult result,
                               int first, int last) {
        for (int index = first; index < last; index++) {
            float input = value(gate, index), sigmoid = sigmoid(input), activated = input * sigmoid;
            float derivative = sigmoid + activated * (1.0f - sigmoid), upstream = value(gradient, index);
            result.gateGradient().memory().set(index, (upstream * value(up, index)) * derivative);
            result.upGradient().memory().set(index, upstream * activated);
        }
    }

    private static float sigmoid(float value) { return 1.0f / (1.0f + (float) Math.exp(-value)); }
    private static float value(Tensor tensor, int index) {
        return tensor.memory().get(tensor.layout().addressAt(index));
    }

    private static void materialize(Tensor... tensors) {
        for (Tensor tensor : tensors) {
            tensor.materialize();
        }
    }
}
