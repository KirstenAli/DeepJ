package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.autograd.GradientState;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class StorageAliasesTest {
    @Test
    void liveViewsScanPastClosedScopesAndPreserveExternalRegistrations() {
        var aliases = new StorageAliases(new GradientState());
        var scoped = new GradientState();
        try (var ignored = TensorScope.open()) { aliases.add(scoped); }
        assertFalse(aliases.liveViews());
        var external = new GradientState();
        var reference = aliases.add(external);
        assertTrue(aliases.liveViews());
        reference.clear();
        assertFalse(aliases.liveViews());
        assertEquals(1, aliases.trackedViews());
    }

    @Test
    void savedViewsScanPastUnusedStatesUntilTheirBackwardGraphIsReleased() {
        var aliases = new StorageAliases(new GradientState());
        var unused = new GradientState();
        aliases.add(unused);
        Tensor view = Tensor.ones(1, 2).requiresGrad(true);
        aliases.add(view.gradientState());
        Tensor result = view.sumTensor();
        assertTrue(aliases.savedForBackward());
        result.backward();
        assertFalse(aliases.savedForBackward());
    }

    @Test
    void readOnlyViewCreationPrunesDeadRegistrationsWithoutDeviceCleanup() {
        var aliases = new StorageAliases(new GradientState());
        for (int index = 0; index < 10000; index++) {
            aliases.add(new GradientState()).clear();
            assertEquals(1, aliases.trackedViews());
        }
        assertFalse(aliases.liveViews());
        assertEquals(0, aliases.trackedViews());
    }

    @Test
    void weakAliasRegistrationsArePrunedWithoutRequiringGarbageCollection() {
        var aliases = new StorageAliases(new GradientState());
        var state = new GradientState();
        var reference = aliases.add(state);
        assertTrue(aliases.liveViews());
        reference.clear();
        assertFalse(aliases.liveViews());
        assertFalse(aliases.savedForBackward());
    }

    @Test
    void rootAndAliasBackwardGraphsAreBothProtected() {
        Tensor root = Tensor.ones(1, 2).requiresGrad(true);
        Tensor view = Tensor.ones(1, 2).requiresGrad(true);
        var aliases = new StorageAliases(root.gradientState());
        aliases.add(view.gradientState());
        Tensor rootResult = root.sumTensor();
        assertTrue(aliases.savedForBackward());
        rootResult.backward();
        assertFalse(aliases.savedForBackward());
        Tensor viewResult = view.sumTensor();
        assertTrue(aliases.savedForBackward());
        viewResult.backward();
        assertFalse(aliases.savedForBackward());
    }

    @Test
    void physicalStorageReleaseRespectsSavedStateAndRetentions() {
        Tensor tensor = Tensor.ones(1, 2).requiresGrad(true);
        assertTrue(tensor.memory().canReleaseTemporary());
        Tensor output = tensor.sumTensor();
        assertFalse(tensor.memory().canReleaseTemporary());
        output.backward();
        assertTrue(tensor.memory().canReleaseTemporary());
        tensor.retainDeviceBuffer();
        assertFalse(tensor.memory().canReleaseTemporary());
        tensor.releaseDeviceBufferRetention();
        assertTrue(tensor.memory().canReleaseTemporary());
    }
}
