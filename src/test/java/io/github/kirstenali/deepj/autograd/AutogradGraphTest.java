package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AutogradGraphTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void automaticallyDifferentiatesComposedOperationsAndRepeatedOperands() {
        Tensor x = scalar(3).requiresGrad(true);
        x.multiply(x).add(x).backward();
        assertEquals(7, x.grad().item());
        assertFalse(x.grad().requiresGrad(), "this engine computes first-order gradients");
    }

    @Test
    void combinesSharedBranchesBeforeApplyingTheirDerivative() {
        Tensor x = scalar(2).requiresGrad(true);
        Tensor shared = x.multiply(x);
        shared.multiplyScalar(3).add(shared.multiplyScalar(4)).backward();
        assertEquals(28, x.grad().item());
    }

    @Test
    void leafGradientsAccumulateAcrossIndependentPassesUntilCleared() {
        Tensor x = scalar(2).requiresGrad(true);
        x.pow(2).backward();
        x.multiplyScalar(3).backward();
        assertEquals(7, x.grad().item());
        x.zeroGrad();
        assertNull(x.grad());
        x.backward(scalar(2));
        assertEquals(2, x.grad().item());
    }

    @Test
    void parameterAndTensorShareOneGradientBuffer() {
        Parameter parameter = new Parameter(scalar(4));
        Tensor buffer = parameter.grad();
        parameter.value.multiplyScalar(2).backward();
        assertSame(buffer, parameter.value.grad());
        assertEquals(2, parameter.grad().item());
        parameter.zeroGrad();
        assertSame(buffer, parameter.grad());
        assertEquals(0, parameter.value.grad().item());
    }

    @Test
    void frozenParametersReceiveNeitherGradientsNorAdamWeightDecay() {
        Parameter frozen = new Parameter(scalar(5));
        frozen.value.requiresGrad(false);
        Tensor input = scalar(2).requiresGrad(true);
        input.multiply(frozen.value).backward();
        assertEquals(5, input.grad().item());
        assertNull(frozen.grad());
        AdamW.defaultAdamW(0.1f).step(List.of(frozen));
        assertEquals(5, frozen.value.item());
        frozen.zeroGrad();
        assertNull(frozen.grad());
    }

    @Test
    void copyTracksButDetachAndCopyConstructorDoNotAliasOrTrack() {
        Tensor x = scalar(2).requiresGrad(true);
        Tensor detached = x.detach();
        detached.set(0, 0, 9);
        assertEquals(2, x.item());
        assertFalse(detached.requiresGrad());
        assertFalse(new Tensor(x).requiresGrad());
        x.copy().multiplyScalar(3).backward();
        assertEquals(3, x.grad().item());
    }

    @Test
    void backwardReleasesSavedValuesAndRejectsASecondTraversal() {
        Tensor x = scalar(2).requiresGrad(true);
        Tensor output = x.pow(2);
        assertTrue(x.gradientState().savedForBackward());
        assertTrue(output.gradientState().savedForBackward());
        output.backward();
        assertFalse(x.gradientState().savedForBackward());
        assertFalse(output.gradientState().savedForBackward());
        assertThrows(IllegalStateException.class, output::backward);
        assertThrows(IllegalStateException.class, () -> output.requiresGrad(false));
    }

    @Test
    void explicitUpstreamIsRequiredForNonScalarOutputs() {
        Tensor input = Tensor.ones(2, 3).requiresGrad(true);
        Tensor output = input.multiplyScalar(2);
        assertThrows(IllegalArgumentException.class, output::backward);
        assertThrows(IllegalArgumentException.class, () -> output.backward(scalar(1)));
        output.backward(Tensor.ones(2, 3));
        assertEquals(12, input.grad().sum());
    }

    @Test
    void deepGraphsUseAnIterativeTraversal() {
        Tensor input = scalar(1).requiresGrad(true);
        Tensor output = input;
        for (int index = 0; index < 10_000; index++) {
            output = output.addScalar(1);
        }
        output.backward();
        assertEquals(1, input.grad().item());
    }

    @Test
    void constantsDoNotRecordHistoryAndCannotRunBackward() {
        Tensor constant = scalar(2).pow(2);
        assertFalse(constant.requiresGrad());
        assertThrows(IllegalStateException.class, constant::backward);
        assertThrows(IllegalStateException.class, () -> Tensor.ones(2, 2).item());
    }

    private static Tensor scalar(float value) { return Tensor.from2D(new float[][]{{value}}); }
}
