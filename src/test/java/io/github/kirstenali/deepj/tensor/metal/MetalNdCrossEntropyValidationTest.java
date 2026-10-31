package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdCrossEntropyValidationTest extends MetalDifferentialTestSupport {
    private final List<Tensor> owners = new ArrayList<>();
    private static final long[] WORDS = {3, 6, 6, 0, 2, 1, 3, 3, 3, 1, 2, 1, 1, 0, 2, 1, 0};

    @Test
    void nativeRejectsAliasesOutputTypesCountsModesAndTruncatedDescriptors() {
        int[] command = command();
        field(command, 2, command[1]);
        field(command, 3, command[2]);
        field(command, 2, buffer(DType.FLOAT16, 2));
        field(command, 3, buffer(DType.FLOAT32, 5));
        field(command, 6, 0);
        field(command, 6, 4);
        field(command, 4, -1);
        field(command, 5, 100);
        field(command, 5, -1);
        invalid(java.util.Arrays.copyOf(command, 6), WORDS);
        invalid(command, java.util.Arrays.copyOf(WORDS, 16));
    }

    @Test
    void nativeIndependentlyValidatesEveryIntegerTargetAndMaskIncludingExcludedRows() {
        int[] command = command();
        word(command, 10, 3);
        word(command, 11, 0);
        word(command, 11, 2);
        word(command, 12, 2);
        word(command, 13, -1);
        word(command, 14, 3);
        word(command, 14, 16_777_217);
        word(command, 15, 2);
        word(command, 16, -1);
        word(command, 12, 0);
        long[] scalar = {0, 1, 1, 0, 1, 1, 0, 0, 1};
        invalid(new int[]{ComputeGraph.OP_LAYOUT_CROSS_ENTROPY, buffer(DType.FLOAT16, 1),
                buffer(DType.FLOAT32, 1), buffer(DType.FLOAT32, 1), 0, 4, 3}, scalar);
    }

    @Test
    void publicInvalidTargetsAndMaskCannotAllocateOutputsOrMutateLogits() {
        Tensor logits = new Tensor(DType.FLOAT16, 2, 1, 3);
        assertThrows(IllegalArgumentException.class, () -> metal.crossEntropy(logits, new int[]{0}));
        assertThrows(IllegalArgumentException.class,
                () -> metal.crossEntropyGradient(logits, new int[]{0, 3}, new boolean[]{true, false}));
        assertThrows(IllegalArgumentException.class, () -> metal.crossEntropyLoss(logits, new int[]{0, 1}, null));
        assertThrows(IllegalArgumentException.class,
                () -> metal.crossEntropy(logits, new int[]{0, 1}, new boolean[2]));
        assertThrows(IllegalArgumentException.class, () -> metal.crossEntropy(logits, new int[]{0, 1}, null));
        assertNull(logits.getGpuTag());
        assertEquals(0, logits.gradientState().version());
    }

    @Test
    void lossOnlyDistinguishesNeighboringIntegerClassesBeyondFloatPrecision() {
        Tensor logits = new Tensor(DType.FLOAT16, 16_777_218);
        logits.setAt(9, 16_777_216);
        logits.setAt(-9, 16_777_217);
        float high = metal.crossEntropyLoss(logits, new int[]{16_777_216});
        float low = metal.crossEntropyLoss(logits, new int[]{16_777_217});
        assertEquals(18, low - high, 2e-6f);
        assertEquals(Math.log(16_777_216 + Math.exp(9) + Math.exp(-9)) + 9, low, 3e-6);
    }

    private int[] command() {
        return new int[]{ComputeGraph.OP_LAYOUT_CROSS_ENTROPY, buffer(DType.FLOAT16, 6),
                buffer(DType.FLOAT32, 2), buffer(DType.FLOAT32, 6), 0, 10, 3};
    }
    private int buffer(DType dtype, int count) {
        Tensor owner = new Tensor(dtype, count);
        owners.add(owner);
        int id = metal.graph.ensureStorageBuffer(owner).id;
        metal.graph.flush();
        return id;
    }
    private static void field(int[] original, int index, int value) {
        int[] command = original.clone();
        command[index] = value;
        invalid(command, WORDS);
    }
    private static void word(int[] command, int index, long value) {
        long[] words = WORDS.clone();
        words[index] = value;
        invalid(command, words);
    }
    private static void invalid(int[] command, long[] words) {
        assertThrows(RuntimeException.class,
                () -> MetalNative.nativeFlushLayoutOps(command, command.length, words));
    }
}
