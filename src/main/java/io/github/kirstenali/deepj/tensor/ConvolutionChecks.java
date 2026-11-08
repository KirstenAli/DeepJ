package io.github.kirstenali.deepj.tensor;

/** Shape and precision contract for same-padded NCHW/OIHW convolution. */
public final class ConvolutionChecks {
    private ConvolutionChecks() {}

    public static boolean samePadding(Tensor input, Tensor weight, int groups) {
        if (input.rank() != 4 || weight.rank() != 4) return false;
        if (input.numel() == 0 || weight.numel() == 0 || input.dtype() != weight.dtype()) return false;
        return channels(input, weight, groups) && weight.size(2) % 2 == 1 && weight.size(3) % 2 == 1;
    }

    private static boolean channels(Tensor input, Tensor weight, int groups) {
        return groups > 0 && (long) weight.size(1) * groups == input.size(1)
                && weight.size(0) % groups == 0;
    }

    public static void requireSamePadding(Tensor input, Tensor weight, int groups) {
        if (!samePadding(input, weight, groups)) {
            throw new IllegalArgumentException("Expected matching NCHW input and odd OIHW kernels");
        }
    }
}
