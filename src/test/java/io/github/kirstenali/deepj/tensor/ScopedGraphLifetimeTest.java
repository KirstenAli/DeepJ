package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ScopedGraphLifetimeTest extends ComputeGraphTestSupport {
    @Test
    void oneHundredIterationsPlateauWithAllViewObjectsStillStronglyReachable() {
        List<Tensor> aliases = new ArrayList<>();
        Tensor persistent = output().retainDeviceBuffer();
        for (int iteration = 0; iteration < 100; iteration++) {
            try (var ignored = TensorScope.open()) { aliases.add(Tensor.aliasStorage(output())); }
            graph.releaseTemporary();
            assertEquals(1, graph.allocatedBufferIds.size());
            assertNotNull(persistent.getGpuTag());
        }
        assertEquals(100, aliases.size());
        assertTrue(aliases.stream().allMatch(value -> value.getGpuTag() == null));
        assertTrue(runtime.downloads.isEmpty());
    }

    @Test
    void callerOwnedAliasSurvivesCleanupAndPrivateDerivativeRetirement() {
        Tensor publicView;
        try (var scope = TensorScope.open()) {
            publicView = scope.escape(Tensor.aliasStorage(output()));
        }
        GpuBuffer buffer = (GpuBuffer) publicView.getGpuTag();
        graph.retireBackwardBuffer(buffer);
        graph.releaseTemporary();
        assertSame(buffer, publicView.getGpuTag());
        assertTrue(runtime.releaseCalls.isEmpty());
    }

    private Tensor output() { return graph.createOutputTensor(graph.newOutputBuffer(1, 4)); }
}
