package io.github.kirstenali.deepj.tensor;

/** Fused dense ND FLOAT32 SwiGLU, preserving the existing scalar arithmetic order. */
final class CpuDenseSwiGlu {
    private CpuDenseSwiGlu() {}

    static void forward(Tensor gate, Tensor up, Tensor output) {
        var a = CpuFloatSpan.of(gate);
        var b = CpuFloatSpan.of(up);
        var result = CpuFloatSpan.of(output);
        CpuDenseElementwise.ranges(output.numel(), (first, last) -> forwardValues(a, b, result, first, last));
    }

    private static void forwardValues(CpuFloatSpan gate, CpuFloatSpan up, CpuFloatSpan output, int first, int last) {
        for (int index = first; index < last; index++) {
            float value = gate.get(index);
            output.set(index, value * sigmoid(value) * up.get(index));
        }
    }

    static void backward(Tensor gradient, Tensor gate, Tensor up, SwiGluBackwardResult result) {
        var dy = CpuFloatSpan.of(gradient);
        var a = CpuFloatSpan.of(gate);
        var b = CpuFloatSpan.of(up);
        var da = CpuFloatSpan.of(result.gateGradient());
        var db = CpuFloatSpan.of(result.upGradient());
        CpuDenseElementwise.ranges(gradient.numel(),
                (first, last) -> backwardValues(dy, a, b, da, db, first, last));
    }

    private static void backwardValues(CpuFloatSpan gradient, CpuFloatSpan gate, CpuFloatSpan up,
                                       CpuFloatSpan gateGradient, CpuFloatSpan upGradient, int first, int last) {
        for (int index = first; index < last; index++) {
            float value = gate.get(index), s = sigmoid(value), activated = value * s;
            float derivative = s + activated * (1.0f - s), upstream = gradient.get(index);
            gateGradient.set(index, (upstream * up.get(index)) * derivative);
            upGradient.set(index, upstream * activated);
        }
    }

    private static float sigmoid(float value) { return 1.0f / (1.0f + (float) Math.exp(-value)); }
}
