package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;

/** Dense causal softmax uses native vector exponentials with the scalar Java fallback at rounding boundaries. */
public final class CpuNativeAttention {
    private CpuNativeAttention() {}

    public static boolean softmax(Tensor input, Tensor output, int width, float scale) {
        return supported(input, output, width)
                && CpuBlas.causalSoftmax(input.rawData(), output.rawData(), input.matrixRows(), width, scale,
                    workers(input.numel()));
    }

    private static int workers(int elements) {
        if (!DeepJExecutor.isParallelEnabled()) return 1;
        return Math.max(1, Math.min(DeepJExecutor.getNumThreads(), elements / 16384));
    }

    private static boolean supported(Tensor input, Tensor output, int width) {
        return CpuBlas.AVAILABLE && input.numel() >= 4096 && input.numel() <= 262144
                && width == input.size(-1) && width > 0 && CpuNdElementwise.sameMatrices(input, output);
    }
}
