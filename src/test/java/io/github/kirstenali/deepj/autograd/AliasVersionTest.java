package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

final class AliasVersionTest {
    @BeforeEach void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void readOnlyAliasCreationDoesNotAccumulateDeadWeakRegistrations() {
        var version = new MutationVersion(0);
        for (int index = 0; index < 10000; index++) {
            version.track(new GradientState()).clear();
            assertEquals(1, version.trackedStates());
        }
        assertFalse(version.any(GradientState::required));
        assertEquals(0, version.trackedStates());
    }

    @Test
    void aliasesShareVersionsButNotGradientBuffersOrTrackingFlags() {
        var original = new GradientState();
        original.beforeMutation();
        var first = original.newAlias();
        var second = first.newAlias();
        assertEquals(1, second.version());
        first.bindGradient(Tensor.ones(1, 1));
        first.require(true);
        assertFalse(original.required());
        assertFalse(second.required());
        assertNull(original.gradient());
        assertNull(second.gradient());
        assertEquals(1, first.gradient().item());
        assertThrows(IllegalStateException.class, second::beforeMutation);
    }

    @Test
    void managedNoGradWritesThroughAnyAliasInvalidateEverySnapshot() {
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        var alias = input.gradientState().newAlias();
        Tensor output = input.pow(2);
        assertTrue(alias.storageSavedForBackward());
        assertFalse(alias.savedForBackward());
        try (var ignored = GradMode.noGrad()) {
            alias.beforeMutation();
        }
        assertEquals(input.gradientState().version(), alias.version());
        assertThrows(IllegalStateException.class, output::backward);
        assertNull(input.grad());
        assertFalse(alias.storageSavedForBackward());
    }

    @Test
    void creatingAliasesDoesNotInvalidateAnExistingBackwardGraph() {
        Tensor input = Tensor.ones(1, 1).requiresGrad(true);
        Tensor output = input.pow(2);
        var alias = input.gradientState().newAlias();
        assertTrue(input.gradientState().storageSavedForBackward());
        output.backward();
        assertEquals(2, input.grad().item());
        assertNull(alias.gradient());
        assertFalse(alias.storageSavedForBackward());
        assertFalse(input.gradientState().storageSavedForBackward());
    }

    @Test
    void untrackedAliasesAllowManagedMutationAndKeepOneCounter() {
        var original = new GradientState();
        var alias = original.newAlias();
        var other = alias.newAlias();
        other.beforeMutation();
        original.beforeMutation();
        assertEquals(2, alias.version());
        assertEquals(alias.version(), other.version());
        assertEquals(alias.version(), original.version());
    }

    @Test
    void clearedWeakRegistrationsDoNotRetainTrackingOrBackwardState() {
        var version = new MutationVersion(17);
        var state = new GradientState();
        state.require(true);
        var reference = version.track(state);
        assertTrue(version.any(GradientState::required));
        reference.clear();
        assertFalse(version.any(GradientState::required));
        version.beforeMutation();
        assertEquals(18, version.value());
    }
}
