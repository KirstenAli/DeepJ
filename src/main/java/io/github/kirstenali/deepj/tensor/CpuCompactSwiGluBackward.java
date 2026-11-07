package io.github.kirstenali.deepj.tensor;

/** Compact saved activations feed bounded native tiles; both adjoints stay FLOAT32. */
final class CpuCompactSwiGluBackward {
    private CpuCompactSwiGluBackward() {}

    static boolean apply(Tensor gradient, Tensor gate, Tensor up, SwiGluBackwardResult output) {
        if (!CpuCompactPointwise.supports(gate, up) || !gradient.isContiguous()) return false;
        CpuDenseElementwise.ranges(gradient.numel(), (first, last) ->
                tile(gradient, gate, up, output, first, last));
        return true;
    }

    private static void tile(Tensor gradient, Tensor gate, Tensor up, SwiGluBackwardResult output,
                              int first, int last) {
        var dy = new CpuProductValues(gradient);
        var a = new CpuProductValues(gate);
        var b = new CpuProductValues(up);
        int[] shape = {gradient.storageOffset() + first, gate.storageOffset() + first, up.storageOffset() + first,
            first, last - first, dy.type(), a.type(), b.type()};
        boolean completed = CpuBlas.compactSwiGluBackward(dy.storage(), a.storage(), b.storage(),
                output.gateGradient().memory().floats(), output.upGradient().memory().floats(), shape);
        if (!completed) CpuNdSwiGlu.backwardRange(gradient, gate, up, output, first, last);
    }
}
