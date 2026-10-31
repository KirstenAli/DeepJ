package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.loss.NdCrossEntropyReference;
import io.github.kirstenali.deepj.tensor.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import java.util.Arrays;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class MetalNdCrossEntropyTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @EnumSource(DType.class)
    void ranksOneThroughSixAndStridesMatchIndependentDoubleReferences(DType dtype) {
        for (int rank = 1; rank <= 6; rank++) {
            int[] shape = new int[rank];
            Arrays.fill(shape, 2);
            shape[rank - 1] = 7;
            Tensor logits = Tensor.from(values((1 << (rank - 1)) * 7), dtype, shape);
            reference(logits, false);
            reference(logits, true);
            if (rank > 1) reference(logits.transpose(0, rank - 1), true);
        }
    }

    private void reference(Tensor logits, boolean masked) {
        int rows = logits.numel() / logits.size(-1);
        int[] targets = randomTargets(rows, logits.size(-1), 9021);
        boolean[] mask = new boolean[rows];
        for (int row = 0; row < rows; row++) { mask[row] = !masked || row % 3 == 0; }
        var result = masked ? metal.crossEntropy(logits, targets, mask) : metal.crossEntropy(logits, targets);
        assertEquals(NdCrossEntropyReference.loss(logits, targets, mask), result.loss().item(), 3e-6);
        assertArrayEquals(NdCrossEntropyReference.gradient(logits, targets, mask),
                result.gradient().toFloatArray(), 2e-7f);
        assertArrayEquals(logits.shape(), result.gradient().shape());
        assertEquals(DType.FLOAT32, result.gradient().dtype());
        float loss = masked ? metal.crossEntropyLoss(logits, targets, mask) : metal.crossEntropyLoss(logits, targets);
        assertEquals(result.loss().item(), loss);
        Tensor gradient = masked ? metal.crossEntropyGradient(logits, targets, mask)
                : metal.crossEntropyGradient(logits, targets);
        assertArrayEquals(result.gradient().toFloatArray(), gradient.toFloatArray());
    }

    @Test
    void fullFloat32NdUsesTheExactExistingMatrixLossAndGradientBits() {
        Tensor matrix = Tensor.from(values(4 * 257), 4, 257);
        int[] targets = {0, 256, 123, 222};
        var old = metal.crossEntropy(matrix, targets);
        for (int[] shape : new int[][]{{4, 1, 257}, {1, 2, 2, 1, 1, 257}}) {
            var result = metal.crossEntropy(matrix.reshape(shape), targets);
            assertEquals(old.loss().item(), result.loss().item());
            assertArrayEquals(old.gradient().toFloatArray(), result.gradient().toFloatArray());
            assertEquals(0, result.loss().rank());
        }
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void excludedNonfiniteRowsAreNotReadAndHaveExactZeroGradients(DType dtype) {
        Tensor logits = Tensor.from(new float[]{2, -1, Float.NaN, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.NaN, 1, 3}, dtype, 4, 2);
        int[] targets = {0, 1, 0, 1};
        boolean[] mask = {true, false, false, true};
        referenceMasked(logits, targets, mask);
        referenceMasked(logits.reshape(2, 2, 2), targets, mask);
        assertTrue(Float.isNaN(metal.crossEntropyLoss(logits, targets)));
        assertTrue(Float.isNaN(metal.crossEntropy(logits, targets).loss().item()));
    }

    private void referenceMasked(Tensor logits, int[] targets, boolean[] mask) {
        var actual = metal.crossEntropy(logits, targets, mask);
        var expected = cpu.crossEntropy(logits, targets, mask);
        assertEquals(expected.loss().item(), actual.loss().item(), 1e-6f);
        assertArrayEquals(expected.gradient().toFloatArray(), actual.gradient().toFloatArray(), 1e-7f);
        assertEquals(actual.loss().item(), metal.crossEntropyLoss(logits, targets, mask));
        assertArrayEquals(actual.gradient().toFloatArray(), metal.crossEntropyGradient(logits, targets, mask)
                .toFloatArray());
    }

    @Test
    void maskedFiniteGradientsKeepTheOriginalTwoStageScalingOrder() {
        Tensor logits = Tensor.from(values(15), 5, 3);
        int[] targets = {0, 1, 2, 0, 1};
        boolean[] mask = {true, false, true, false, true};
        Tensor weights = Tensor.from(new float[]{1, 0, 1, 0, 1}, 5, 1);
        Tensor old = metal.multiplyScalar(metal.multiplyBroadcastCols(metal.crossEntropyGradient(logits, targets),
                weights), 5.0f / 3.0f);
        assertArrayEquals(old.toFloatArray(), metal.crossEntropyGradient(logits, targets, mask).toFloatArray());
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void vocabularySizedReductionsKeepFloat32GradientsAndHostStorageLazy(DType dtype) {
        Tensor logits = metal.copy(Tensor.from(values(2 * 16_384), dtype, 2, 1, 16_384));
        var result = metal.crossEntropy(logits, new int[]{16_383, 512}, new boolean[]{true, false});
        metal.graph.flush();
        assertFalse(TensorTestStorage.hasHostStorage(logits));
        assertFalse(TensorTestStorage.hasHostStorage(result.gradient()));
        assertFalse(TensorTestStorage.hasHostStorage(result.loss()));
        assertEquals(2L * 16_384 * dtype.bytes(), logits.storageBytes());
        reference(logits, true);
    }

    private static float[] values(int count) {
        float[] result = new float[count];
        Random random = new Random(11029);
        for (int index = 0; index < count; index++) { result[index] = random.nextFloat() * 18 - 9; }
        return result;
    }
}
