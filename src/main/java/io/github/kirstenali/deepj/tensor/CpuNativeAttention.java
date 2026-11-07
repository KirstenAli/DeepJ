package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.concurrent.DeepJExecutor;
import java.util.function.IntConsumer;

/** Dense causal softmax uses native vector exponentials with the scalar Java fallback at rounding boundaries. */
public final class CpuNativeAttention {
    private CpuNativeAttention() {}

    public static boolean softmax(Tensor input, Tensor output, int width, float scale) {
        return softmax(input, output, width, scale, null);
    }

    public static boolean softmax(Tensor input, Tensor output, int width, float scale, IntConsumer scalarRow) {
        if (!supported(input, output, width)) return false;
        int[] fallback = new int[input.matrixRows()];
        if (!CpuBlas.causalSoftmax(input.rawData(), output.rawData(), input.matrixRows(), width, scale,
                workers(input.numel()), fallback)) return false;
        return completeRows(fallback, scalarRow);
    }

    private static boolean completeRows(int[] fallback, IntConsumer scalarRow) {
        for (int row = 0; row < fallback.length; row++) {
            if (fallback[row] == 0) continue;
            if (scalarRow == null) return false;
            scalarRow.accept(row);
        }
        return true;
    }

    public static boolean compactSoftmax(Tensor input, Tensor output, int width, float scale, boolean causal) {
        if (input.dtype() == DType.FLOAT32 || input.numel() < 256 || !CpuBlas.AVAILABLE) return false;
        if (!compactLayouts(input, output, width)) return false;
        int[] shape = {input.storageOffset(), output.storageOffset(), input.numel() / width, width,
            input.dtype().storageCode(), causal ? 1 : 0, workers(input.numel())};
        int[] fallback = new int[shape[2]];
        boolean nativeResult = CpuBlas.compactSoftmax(input.memory().shorts(), output.memory().shorts(),
                shape, scale, fallback);
        if (!nativeResult) return false;
        fallbackRows(input, output, width, scale, causal, fallback);
        return true;
    }

    private static void fallbackRows(Tensor input, Tensor output, int width, float scale,
                                      boolean causal, int[] fallback) {
        for (int row = 0; row < fallback.length; row++) {
            if (fallback[row] == 0) continue;
            if (causal) CpuNdPosition.softmaxRow(input, output, row, width, scale);
            else CpuNdSoftmax.row(input, output, row * width, width, 1);
        }
    }

    private static boolean compactLayouts(Tensor input, Tensor output, int width) {
        return width >= 1 && width <= 4096 && width == input.size(-1) && input.hasSameShape(output)
                && input.dtype() == output.dtype() && input.isContiguous() && output.isContiguous();
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
