package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetalNdTrainingValidationTest extends MetalCommandValidationSupport {
    private static final long[] WORDS = {2, 6, 6, 0, 2, 3, 3, 1,
            1, 3, 3, 0, 3, 1, 2, 2, 2, 0, 2, 1, 1, 1,
            2, 6, 6, 0, 2, 3, 0, 1};

    @Test
    void forwardRejectsAliasedWronglyTypedOrWronglySizedSavedValuesAndInvalidScalars() {
        int input = buffer(DType.FLOAT16, 6), gain = buffer(DType.FLOAT16, 3);
        int output = buffer(DType.FLOAT16, 6), normalized = buffer(DType.FLOAT32, 6), std = buffer(DType.FLOAT32, 2);
        int[] command = {ComputeGraph.OP_LAYOUT_NORMALIZE, input, gain, gain, output, normalized, std,
                0, 8, 8, Float.floatToRawIntBits(1e-5f), 1};
        invalidField(command, 4, input);
        invalidField(command, 5, output);
        invalidField(command, 6, normalized);
        invalidField(command, 8, 0);
        invalidField(command, 10, Float.floatToRawIntBits(Float.NaN));
        invalidField(command, 10, 0);
        invalidField(command, 11, 2);
        invalidField(command, 4, buffer(DType.FLOAT32, 6));
        invalidField(command, 5, buffer(DType.FLOAT16, 6));
    }

    @Test
    void backwardRejectsWrongSavedShapesDtypesAliasesAndUnknownModes() {
        int gradient = buffer(DType.FLOAT32, 6), normalized = buffer(DType.FLOAT32, 6);
        int std = buffer(DType.FLOAT32, 2), gain = buffer(DType.FLOAT16, 3), output = buffer(DType.FLOAT32, 6);
        int[] command = {ComputeGraph.OP_LAYOUT_NORMALIZATION_BACKWARD, gradient, normalized, std, gain,
                output, 0, 0, 14, 8, 1};
        invalidField(command, 5, normalized);
        invalidField(command, 5, buffer(DType.FLOAT16, 6));
        invalidField(command, 8, 8);
        invalidField(command, 9, 0);
        invalidField(command, 10, -1);
        invalidField(command, 6, 100);
        invalid(new int[]{ComputeGraph.OP_LAYOUT_NORMALIZATION_BACKWARD, gradient, normalized, std, gain}, WORDS);
    }

    @Test
    void swigluAdjointsRequireEqualInputLayoutsAndIndependentFloat32Outputs() {
        int input = buffer(DType.FLOAT16, 6), gradient = buffer(DType.FLOAT32, 6);
        int first = buffer(DType.FLOAT32, 6), second = buffer(DType.FLOAT32, 6);
        int[] command = {ComputeGraph.OP_LAYOUT_SWIGLU_BACKWARD, gradient, input, input, first, second, 0, 0, 0};
        invalidField(command, 4, gradient);
        invalidField(command, 5, first);
        invalidField(command, 5, input);
        invalidField(command, 5, buffer(DType.FLOAT32, 5));
        invalidField(command, 7, 8);
        invalidField(command, 8, -1);
    }

    @Test
    void adamRejectsAliasedCompactMismatchedAndOverlappingStateBeforeDispatch() {
        int w = buffer(DType.FLOAT32, 6), g = buffer(DType.FLOAT32, 6);
        int m = buffer(DType.FLOAT32, 6), v = buffer(DType.FLOAT32, 6);
        int[] command = {ComputeGraph.OP_LAYOUT_ADAMW, w, g, m, v, 0, 0, 0, 0,
                bits(0.01f), bits(0.9f), bits(0.999f), bits(1e-8f), 0, bits(0.1f), bits(0.001f)};
        invalidField(command, 3, w);
        invalidField(command, 2, buffer(DType.FLOAT16, 6));
        invalidField(command, 5, 22);
        invalidField(command, 6, 8);
        invalidField(command, 7, -1);
    }

    @Test
    void publicInvalidMetadataDoesNotMutateStateOrAllocateAnOptimizerOutput() {
        Tensor input = new Tensor(DType.FLOAT16, 2, 1, 3), gain = new Tensor(3);
        assertThrows(IllegalArgumentException.class, () -> metal.normalize(input, gain, null, -1, false));
        assertThrows(IllegalArgumentException.class, () -> metal.normalize(input, gain, gain, 1e-5f, false));
        assertThrows(IllegalArgumentException.class,
                () -> metal.layerNormBackward(input, input, new Tensor(1), 3));
        Tensor w = new Tensor(2, 3), m = new Tensor(2, 3), v = new Tensor(2, 3);
        assertThrows(IllegalArgumentException.class,
                () -> metal.adamWUpdate(w, w, m, v, 0.01f, 0.9f, 0.999f, 1e-8f, 0, 0.1f, 0.001f));
        assertEquals(0, w.gradientState().version());
        assertNull(w.getGpuTag());
    }

    @Override protected long[] words() { return WORDS; }
}
