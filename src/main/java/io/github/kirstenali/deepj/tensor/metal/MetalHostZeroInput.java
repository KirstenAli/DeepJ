package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.GpuBuffer;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Arrays;

/** Checks CPU values directly before replacing a zero upload with scalar addition. */
final class MetalHostZeroInput {
    private static final float[] ZERO_BLOCK = new float[16_384];

    private MetalHostZeroInput() {}

    static boolean canSkip(Tensor target, Tensor source) {
        return source.getGpuTag() instanceof GpuBuffer && isUnboundZero(target);
    }

    static boolean isUnboundZero(Tensor target) {
        return target.getGpuTag() == null
                && target.rawData() != null
                && positiveZeros(target.rawData());
    }

    static boolean positiveZeros(float[] values) {
        for (int offset = 0; offset < values.length; offset += ZERO_BLOCK.length) {
            int length = Math.min(ZERO_BLOCK.length, values.length - offset);
            if (Arrays.mismatch(values, offset, offset + length, ZERO_BLOCK, 0, length) != -1) return false;
        }
        return true;
    }
}
