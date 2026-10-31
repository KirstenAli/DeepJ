package io.github.kirstenali.deepj.loss;

import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;
import static org.junit.jupiter.api.Assertions.*;

class NdCrossEntropyTest {
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void ranksOneThroughSixMatchIndependentReferencesInEveryDtype() {
        Random random = new Random(9107);
        for (DType dtype : DType.values()) {
            for (int rank = 1; rank <= 6; rank++) {
                checkRandomCase(random, dtype, rank);
            }
        }
    }

    private static void checkRandomCase(Random random, DType dtype, int rank) {
        int[] shape = new int[rank];
        Arrays.fill(shape, 2);
        shape[rank - 1] = 5;
        int count = 1 << (rank - 1);
        Tensor logits = Tensor.from(randomValues(random, count * 5), dtype, shape);
        int[] targets = new int[count];
        boolean[] mask = new boolean[count];
        for (int row = 0; row < count; row++) {
            targets[row] = random.nextInt(5);
            mask[row] = row % 3 == 0;
        }
        checkReference(logits, targets, mask);
        if (rank > 2) checkReference(logits.transpose(0, 1), targets, mask);
    }

    private static float[] randomValues(Random random, int count) {
        float[] result = new float[count];
        for (int index = 0; index < count; index++) {
            result[index] = random.nextFloat() * 20 - 10;
        }
        return result;
    }

    private static void checkReference(Tensor logits, int[] targets, boolean[] mask) {
        CrossEntropyResult result = Tensor.backend().crossEntropy(logits, targets, mask);
        assertEquals(NdCrossEntropyReference.loss(logits, targets, mask), result.loss().item(), 3e-6);
        assertArrayEquals(NdCrossEntropyReference.gradient(logits, targets, mask),
                result.gradient().toFloatArray(), 1e-7f);
        assertEquals(DType.FLOAT32, result.gradient().dtype());
        assertEquals(DType.FLOAT32, result.loss().dtype());
        assertArrayEquals(logits.shape(), result.gradient().shape());
        assertEquals(result.loss().item(), CrossEntropyLoss.loss(logits, targets, mask));
        assertArrayEquals(result.gradient().toFloatArray(), CrossEntropyLoss.gradient(logits, targets, mask)
                .toFloatArray());
    }

    @Test
    void ndAndStridedResultsRetainTheOriginalMatrixArithmeticOrder() {
        Tensor logits = Tensor.from(new float[]{0.2f, -0.4f, 0.7f, 1.1f, 0.3f, -0.2f,
                5, -7, 3, 1000, 999, 1001}, 2, 2, 3);
        int[] targets = {2, 0, 1, 2};
        checkMatrixOrder(logits, targets);
        checkMatrixOrder(logits.permute(1, 0, 2), targets);
        checkMatrixOrder(logits.to(DType.FLOAT16), targets);
    }

    private static void checkMatrixOrder(Tensor logits, int[] targets) {
        Tensor matrix = Tensor.from(logits.toFloatArray(), targets.length, logits.size(-1));
        assertEquals(CrossEntropyLoss.loss(matrix, targets), CrossEntropyLoss.loss(logits, targets));
        assertArrayEquals(CrossEntropyLoss.gradient(matrix, targets).toFloatArray(),
                CrossEntropyLoss.gradient(logits, targets).toFloatArray());
        boolean[] mask = {true, false, true, true};
        assertEquals(CrossEntropyLoss.loss(matrix, targets, mask), CrossEntropyLoss.loss(logits, targets, mask));
        assertArrayEquals(CrossEntropyLoss.gradient(matrix, targets, mask).toFloatArray(),
                CrossEntropyLoss.gradient(logits, targets, mask).toFloatArray());
    }

    @Test
    void maskedAutogradAndTensorGridTargetsMatchFiniteDifferences() {
        Tensor logits = Tensor.from(new float[]{0.2f, -0.4f, 0.7f, 1.1f, 0.3f, -0.2f}, 1, 2, 3);
        int[] targets = {2, 0};
        assertTensorGradients(logits, value -> CrossEntropyLoss.forward(value, targets, new boolean[]{false, true}),
                Tensor.from(new float[]{0.7f}, new int[0]), 1e-3f, 1e-4f, 1e-3f);
        Tensor grid = Tensor.from(new float[]{2, 0}, 1, 2);
        var loss = new CrossEntropyLoss();
        assertEquals(CrossEntropyLoss.loss(logits, targets), loss.loss(logits, grid));
        assertArrayEquals(CrossEntropyLoss.gradient(logits, targets).toFloatArray(),
                loss.gradient(logits, grid).toFloatArray());
        logits.requiresGrad(true);
        loss.forward(logits, grid).multiplyScalar(2).backward();
        assertArrayEquals(CrossEntropyLoss.gradient(logits, targets).multiplyScalar(2).toFloatArray(),
                logits.grad().toFloatArray());
    }

    @Test
    void matrixScalarShapesDoNotDependOnDtypeOrLayout() {
        for (DType dtype : DType.values()) {
            Tensor matrix = Tensor.from(new float[]{1, 2, 3, 4}, dtype, 2, 2).requiresGrad(true);
            Tensor loss = CrossEntropyLoss.forward(matrix, new int[]{0, 1}, null);
            assertArrayEquals(new int[]{1, 1}, loss.shape());
            assertArrayEquals(new int[]{1, 1}, matrix.sumTensor().shape());
            assertArrayEquals(new int[]{1, 1}, matrix.transpose(0, 1).sumTensor().shape());
            loss.backward();
            assertEquals(DType.FLOAT32, matrix.grad().dtype());
        }
        Tensor vector = Tensor.from(new float[]{1, 2});
        assertEquals(0, CrossEntropyLoss.forward(vector, new int[]{1}, null).rank());
    }

    @Test
    void invalidShapesTargetsAndMasksAreRejectedBeforeComputing() {
        Tensor logits = new Tensor(2, 2, 3);
        int[] targets = {0, 1, 2, 0};
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(null, targets));
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(new Tensor(new int[0]), targets));
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(new Tensor(2, 0), targets));
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(new Tensor(0, 3), new int[0]));
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(logits, (int[]) null));
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(logits, new int[]{0}));
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(logits, new int[]{0, 1, -1, 0}));
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(logits, new int[]{0, 1, 3, 0}));
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(logits, targets, new boolean[4]));
        assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.loss(logits, targets, new boolean[1]));
        assertThrows(IllegalArgumentException.class, () -> Tensor.backend().crossEntropy(logits, targets, null));
    }

    @Test
    void tensorTargetsMustBeFiniteIntegerClassIds() {
        for (float invalid : new float[]{0.1f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
            Tensor target = Tensor.from(new float[]{invalid}, new int[0]);
            assertThrows(IllegalArgumentException.class, () -> CrossEntropyLoss.toIntTargets(target));
        }
        Tensor source = Tensor.from(new float[]{0, 1, 2, 3}, 2, 2).transpose(0, 1);
        assertArrayEquals(new int[]{0, 2, 1, 3}, CrossEntropyLoss.toIntTargets(source));
    }
}
