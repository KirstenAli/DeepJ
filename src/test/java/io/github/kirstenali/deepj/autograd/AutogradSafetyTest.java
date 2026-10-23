package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorAdapters;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;

class AutogradSafetyTest {
    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void rejectsInPlaceUpdatesToTrackedTensorsAndTrackedOperands() {
        Tensor input = Tensor.ones(1, 2).requiresGrad(true);
        assertThrows(IllegalStateException.class, () -> input.multiplyScalarInPlace(2));
        assertThrows(IllegalStateException.class, () -> Tensor.ones(1, 2).addInPlace(input));
        assertThrows(IllegalStateException.class, () -> Tensor.ones(1, 2).setRow(0, input, 0));
        assertThrows(IllegalStateException.class,
                () -> Tensor.scatterAddRows(Tensor.ones(1, 2), new int[]{0}, input));
        assertThrows(IllegalStateException.class, () -> input.set(0, 0, 3));
    }

    @Test
    void detectsManagedChangesToSavedParametersEvenInsideNoGrad() {
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        Tensor output = input.pow(2);
        try (var ignored = GradMode.noGrad()) {
            input.set(0, 0, 4);
        }
        assertThrows(IllegalStateException.class, output::backward);
        assertNull(input.grad());
    }

    @Test
    void detectsChangesToSavedConstantsAndBulkWrites() {
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        Tensor constant = Tensor.ones(1, 1);
        Tensor output = input.multiply(constant);
        TensorAdapters.unpackF32Into(new float[]{3}, constant);
        assertThrows(IllegalStateException.class, () -> constant.requiresGrad(true));
        assertThrows(IllegalStateException.class, output::backward);
        assertFalse(constant.gradientState().savedForBackward());
        constant.requiresGrad(true);
    }

    @Test
    void detectsRowWritesAndBackendSwitchesBeforeAccumulatingGradients() {
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        Tensor output = input.pow(2);
        Tensor.setBackend(new CpuBackend());
        assertThrows(IllegalStateException.class, output::backward);
        assertNull(input.grad());
        Tensor fresh = input.pow(2);
        try (var ignored = GradMode.noGrad()) {
            input.setRow(0, Tensor.zeros(1, 1), 0);
        }
        assertThrows(IllegalStateException.class, fresh::backward);
    }

    @Test
    void noGradIsNestedAndRestoresTrackingAfterExceptions() {
        Tensor parameter = Tensor.ones(1, 1).requiresGrad(true);
        assertThrows(IllegalArgumentException.class, () -> {
            try (var first = GradMode.noGrad(); var second = GradMode.noGrad()) {
                assertFalse(parameter.pow(2).requiresGrad());
                parameter.addScalarInPlace(1);
                throw new IllegalArgumentException("test");
            }
        });
        assertTrue(GradMode.isEnabled());
        parameter.pow(2).backward();
        assertEquals(4, parameter.grad().item());
    }

    @Test
    void scopesAreThreadLocalAndCannotBeClosedByAnotherThread() {
        try (var scope = GradMode.noGrad()) {
            assertTrue(CompletableFuture.supplyAsync(GradMode::isEnabled).join());
            assertTrue(CompletableFuture.supplyAsync(() -> rejectsForeignClose(scope)).join());
            assertFalse(GradMode.isEnabled());
        }
        assertTrue(GradMode.isEnabled());
    }

    private static boolean rejectsForeignClose(GradMode scope) {
        assertThrows(IllegalStateException.class, scope::close);
        return true;
    }

    @Test
    void scopesRejectOutOfOrderCloseAndAllowRepeatedClose() {
        var outer = GradMode.noGrad();
        var inner = GradMode.noGrad();
        assertThrows(IllegalStateException.class, outer::close);
        inner.close();
        inner.close();
        assertFalse(GradMode.isEnabled());
        outer.close();
        assertTrue(GradMode.isEnabled());
    }

    @Test
    void customRulesRejectMissingWrongShapeOrWrongCountGradients() {
        assertInvalidRule(g -> new Tensor[]{null});
        assertInvalidRule(g -> new Tensor[]{Tensor.ones(2, 2)});
        assertInvalidRule(g -> new Tensor[0]);
        assertTrue(GradMode.isEnabled());
    }

    private static void assertInvalidRule(BackwardRule rule) {
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        Tensor output = Autograd.record(Tensor.ones(1, 1), rule, input);
        assertThrows(RuntimeException.class, output::backward);
        assertFalse(input.gradientState().savedForBackward());
    }

    @Test
    void customOperationsCannotCreateSelfCycles() {
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        assertThrows(IllegalArgumentException.class,
                () -> Autograd.record(input, g -> new Tensor[]{g}, input));
    }

    @Test
    void customOperationsCannotOverwriteAnExistingGraphOrCreateAncestorCycles() {
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        Tensor output = input.pow(2);
        assertThrows(IllegalArgumentException.class, () -> Autograd.record(input, g -> new Tensor[]{g}, output));
        assertThrows(IllegalArgumentException.class, () -> Autograd.record(output, g -> new Tensor[]{g}, input));
        Tensor constant = Tensor.ones(1, 1);
        assertThrows(IllegalArgumentException.class,
                () -> Autograd.record(constant, g -> new Tensor[]{g, g}, constant, input));
        output.backward();
        assertEquals(2, input.grad().item());
    }

    @Test
    void validatesSavedOutputsAsWellAsMetadataOnlyInputs() {
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        Tensor output = input.exp();
        try (var ignored = GradMode.noGrad()) {
            output.addScalarInPlace(1);
        }
        assertThrows(IllegalStateException.class, output::backward);
        assertNull(input.grad());
        assertFalse(output.gradientState().savedForBackward());
    }

    @Test
    void parameterRejectsTrackedOrIncorrectlyShapedGradientBuffers() {
        Parameter parameter = new Parameter(Tensor.ones(1, 1));
        assertThrows(IllegalArgumentException.class, () -> parameter.setGradient(Tensor.ones(2, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> parameter.setGradient(Tensor.ones(1, 1).requiresGrad(true)));
        parameter.setGradient(null);
        assertNull(parameter.grad());
    }
}
