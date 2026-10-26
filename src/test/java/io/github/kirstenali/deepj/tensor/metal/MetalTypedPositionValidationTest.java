package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MetalTypedPositionValidationTest extends MetalCommandValidationSupport {
    private static final long[] WORDS = {2, 16, 16, 0, 4, 4, 4, 1, 2, 8, 8, 0, 4, 2, 2, 1};

    @Test
    void nativeCausalRequiresLogicalSquaresMatchingDtypeDistinctStorageAndValidMode() {
        int source = buffer(DType.FLOAT16, 16), output = buffer(DType.FLOAT16, 16);
        int[] command = {ComputeGraph.OP_LAYOUT_CAUSAL, source, output, 0, 4, bits(0.5f), 2};
        invalidField(command, 2, source);
        invalidField(command, 2, buffer(DType.FLOAT32, 16));
        invalidField(command, 2, buffer(DType.FLOAT16, 15));
        invalidField(command, 3, 8);
        invalidField(command, 3, -1);
        invalidField(command, 4, 0);
        invalidField(command, 4, 3);
        invalidField(command, 5, bits(Float.NaN));
        invalidField(command, 6, 0);
        invalidField(command, 6, 3);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_CAUSAL, source, output});
    }

    @Test
    void nativeRotaryRequiresMatchingTablesIndependentOutputAndBooleanInverse() {
        int input = buffer(DType.FLOAT16, 16), cosine = buffer(DType.FLOAT32, 8);
        int sine = buffer(DType.BFLOAT16, 8), output = buffer(DType.FLOAT16, 16);
        int[] command = {ComputeGraph.OP_LAYOUT_ROTARY, input, cosine, sine, output, 0, 8, 8, 4, 0};
        invalidField(command, 4, input);
        invalidField(command, 4, buffer(DType.FLOAT32, 16));
        invalidField(command, 4, buffer(DType.FLOAT16, 15));
        invalidField(command, 5, 8);
        invalidField(command, 6, 0);
        invalidField(command, 7, -1);
        invalidField(command, 8, 0);
        invalidField(command, 8, 3);
        invalidField(command, 8, 8);
        invalidField(command, 9, 2);
    }

    @Test
    void publicEmptyAndMalformedMetadataNeverFallThroughToFloat32OnlyKernels() {
        Tensor input = new Tensor(DType.FLOAT16, 4, 4), table = new Tensor(4, 2);
        assertThrows(IllegalArgumentException.class, () -> metal.causalSoftmax(input, 4, Float.NaN));
        assertThrows(IllegalArgumentException.class, () -> metal.causalMask(input, 3));
        assertThrows(IllegalArgumentException.class, () -> metal.rotary(input, table, table, 0, false));
        assertThrows(IllegalArgumentException.class, () -> metal.graph.layoutCausal(input, 4, 1, 3));
        assertNull(input.getGpuTag());
    }

    @Override protected long[] words() { return WORDS; }
}
