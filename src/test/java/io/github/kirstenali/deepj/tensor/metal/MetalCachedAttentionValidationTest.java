package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.DType;
import org.junit.jupiter.api.Test;

class MetalCachedAttentionValidationTest extends MetalCommandValidationSupport {
    @Override protected long[] words() { return new long[0]; }

    @Test void malformedSoftcapsAndLocalWindowsAreRejectedBeforeDispatch() {
        int[] original = command(), command = java.util.Arrays.copyOf(original, 13);
        command[0] = ComputeGraph.OP_CACHED_ATTENTION_SOFTCAP;
        command[11] = bits(50);
        command[12] = 7;
        invalidField(command, 11, bits(-1));
        invalidField(command, 11, bits(Float.NaN));
        invalidField(command, 11, bits(Float.POSITIVE_INFINITY));
        invalidField(command, 12, -1);
        invalidField(command, 12, command[9]);
        invalidField(command, 6, 257);
    }

    @Test void malformedShapesAndScalesAreRejectedBeforeDispatch() {
        int[] command = command();
        for (int field : new int[]{5, 6, 7, 8, 9}) { invalidField(command, field, 0); }
        invalidField(command, 5, 3);
        invalidField(command, 6, 129);
        invalidField(command, 9, 513);
        invalidField(command, 9, 33);
        invalidField(command, 10, bits(Float.NaN));
    }

    @Test void wrongStorageAndAliasedOutputsAreRejected() {
        int[] command = command();
        invalidField(command, 1, buffer(DType.FLOAT32, 4 * 64));
        invalidField(command, 2, buffer(DType.BFLOAT16, 2 * 32 * 64));
        invalidField(command, 4, buffer(DType.FLOAT16, 4 * 64 - 1));
        for (int field : new int[]{1, 2, 3}) { invalidField(command, 4, command[field]); }
    }

    private int[] command() {
        int q = buffer(DType.FLOAT16, 4 * 64), k = buffer(DType.FLOAT16, 2 * 32 * 64);
        int v = buffer(DType.FLOAT16, 2 * 32 * 64), output = buffer(DType.FLOAT16, 4 * 64);
        return new int[]{ComputeGraph.OP_CACHED_ATTENTION, q, k, v, output, 4, 64, 2, 32, 20, bits(.125f)};
    }
}
