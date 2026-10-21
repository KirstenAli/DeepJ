package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.loss.MSELoss;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class LossGradientTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void crossEntropyAndItsScaledMaskedVersionMatchNumericalGradients() {
        Tensor logits = point(3, 4);
        int[] targets = {2, 1, 0};
        OperationGradientAssertions.check(x -> CrossEntropyLoss.forward(x, targets, null), logits);
        OperationGradientAssertions.check(x -> CrossEntropyLoss.forward(x, targets,
                new boolean[]{false, true, true}).multiplyScalar(2), logits);
    }

    @Test
    void maskedRowsAreExactlyZeroAndInputsAreSnapshotted() {
        Tensor logits = point(3, 4).requiresGrad(true);
        int[] targets = {2, 1, 0};
        boolean[] mask = {false, true, true};
        Tensor expected = CrossEntropyLoss.gradient(logits, targets, mask);
        Tensor loss = CrossEntropyLoss.forward(logits, targets, mask);
        targets[1] = 3;
        mask[0] = true;
        loss.backward();
        assertArrayEquals(expected.data, logits.grad().data, 1e-6f);
    }

    @Test
    void supervisedCrossEntropyReturnsADifferentiableScalar() {
        Tensor logits = point(3, 4).requiresGrad(true);
        Tensor targets = CrossEntropyLoss.fromIntTargets(new int[]{2, 1, 0});
        Tensor loss = new CrossEntropyLoss().forward(logits, targets);
        assertEquals(CrossEntropyLoss.loss(logits, new int[]{2, 1, 0}), loss.item(), 1e-6f);
        loss.backward();
        assertNotNull(logits.grad());
        assertNull(targets.grad());
    }

    @Test
    void mseDifferentiatesEitherOperandAndPreservesMeanReduction() {
        Tensor prediction = point(2, 3), target = Tensor.ones(2, 3);
        MSELoss loss = new MSELoss();
        OperationGradientAssertions.check(x -> loss.forward(x, target), prediction);
        OperationGradientAssertions.check(x -> loss.forward(prediction, x), target);
        assertEquals(loss.loss(prediction, target), loss.forward(prediction, target).item(), 1e-6f);
    }

    @Test
    void rejectsInvalidMasksInTheDifferentiableEntryPoint() {
        Tensor logits = point(3, 4).requiresGrad(true);
        int[] targets = {2, 1, 0};
        assertThrows(IllegalArgumentException.class,
                () -> CrossEntropyLoss.forward(logits, targets, new boolean[3]));
        assertThrows(IllegalArgumentException.class,
                () -> CrossEntropyLoss.forward(logits, targets, new boolean[2]));
    }

    private static Tensor point(int rows, int columns) {
        return Tensor.random(rows, columns, new Random(411));
    }
}
