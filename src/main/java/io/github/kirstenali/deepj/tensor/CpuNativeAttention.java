package io.github.kirstenali.deepj.tensor;

/** Dense causal softmax uses native vector exponentials with the scalar Java fallback at rounding boundaries. */
public final class CpuNativeAttention {
    private CpuNativeAttention() {}

    public static boolean softmax(Tensor input, Tensor output, int width, float scale) {
        return supported(input, output, width)
                && CpuBlas.causalSoftmax(input.rawData(), output.rawData(), input.matrixRows(), width, scale);
    }

    private static boolean supported(Tensor input, Tensor output, int width) {
        return CpuBlas.AVAILABLE && input.numel() >= 4096 && input.numel() <= 262144
                && width == input.size(-1) && width > 0 && CpuNdElementwise.sameMatrices(input, output);
    }
}
