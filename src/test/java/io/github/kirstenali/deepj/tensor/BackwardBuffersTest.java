package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class BackwardBuffersTest extends ComputeGraphTestSupport {
    @Test
    void anEarlierReleasedScratchLeaseCannotRetireANewOwnerOfTheSameId() {
        Tensor replacement;
        try (var scope = BackwardBuffers.begin()) {
            GpuBuffer earlier = scope.capture(() -> graph.newOutputBuffer(1, 4));
            graph.releaseAll();
            GpuBuffer current = graph.newOutputBuffer(1, 4);
            assertEquals(earlier.id, current.id);
            replacement = graph.createOutputTensor(current);
        }
        assertNotNull(replacement.getGpuTag());
        assertTrue(graph.backwardBufferIds.isEmpty());
        graph.releaseAll();
    }

    @Test
    void completedPrivateDerivativesReuseStorageWithoutGarbageCollection() {
        List<Tensor> privateValues;
        GpuBuffer input = graph.ensureGpuBuffer(Tensor.ones(1, 4));
        try (var scope = BackwardBuffers.begin()) {
            privateValues = scope.capture(() -> chain(input));
            assertTrue(graph.backwardBufferIds.isEmpty());
        }
        assertEquals(4, graph.backwardBufferIds.size());
        assertEquals(3, GraphBufferReuse.prepare(graph).allocations().size());
        graph.flush();
        assertEquals(1, graph.allocatedBufferIds.size());
        assertTrue(graph.backwardBufferIds.isEmpty());
        assertTrue(privateValues.stream().allMatch(value -> value.getGpuTag() == null));
    }

    @Test
    void customRuleInputsRetainedAndSavedValuesAreNotRetired() {
        Tensor retained, saved, exposed;
        try (var scope = BackwardBuffers.begin()) {
            retained = scope.capture(() -> output()).retainDeviceBuffer();
            saved = scope.capture(() -> output()).requiresGrad(true).exp();
            exposed = scope.capture(() -> output());
            scope.expose(exposed);
            scope.expose(Tensor.ones(1, 4));
        }
        assertNotNull(retained.getGpuTag());
        assertNotNull(exposed.getGpuTag());
        assertTrue(saved.requiresGrad());
        assertTrue(graph.backwardBufferIds.isEmpty());
    }

    @Test
    void nestingExceptionAndReleaseAllLeaveNoLifetimeRegistrationsBehind() {
        BackwardBuffers outer = BackwardBuffers.begin();
        try (var inner = BackwardBuffers.begin()) {
            assertThrows(IllegalStateException.class, outer::close);
            assertThrows(IllegalArgumentException.class, () -> inner.capture(() -> {
                throw new IllegalArgumentException("rule failed");
            }));
            inner.capture(() -> output());
            graph.releaseAll();
        }
        outer.close();
        outer.close();
        assertTrue(graph.backwardBufferIds.isEmpty());
        output();
        assertTrue(graph.backwardBufferIds.isEmpty());
    }

    @Test
    void scopesCannotBeClosedFromAnotherThread() throws InterruptedException {
        try (var scope = BackwardBuffers.begin()) {
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread thread = new Thread(() -> {
                try {
                    scope.close();
                } catch (Throwable error) {
                    failure.set(error);
                }
            });
            thread.start();
            thread.join();
            assertInstanceOf(IllegalStateException.class, failure.get());
        }
    }

    private Tensor output() { return graph.createOutputTensor(graph.newOutputBuffer(1, 4)); }

    @Test
    void selectiveReleasePreservesUnrelatedCallerOutputsAndRetainedPrivateStorage() {
        Tensor caller = output(), retained, privateValue;
        try (var scope = BackwardBuffers.begin()) {
            privateValue = scope.capture(this::output);
            retained = scope.capture(this::output).retainDeviceBuffer();
            scope.releaseTemporaryResources();
            assertNull(privateValue.getGpuTag());
            assertNotNull(caller.getGpuTag());
            assertNotNull(retained.getGpuTag());
            assertEquals(2, graph.allocatedBufferIds.size());
        }
        assertEquals(2, graph.allocatedBufferIds.size());
    }

    @Test
    void privateDerivativeAliasesDoNotPreventScratchReuseBeforeGarbageCollection() {
        List<Tensor> aliases;
        GpuBuffer input = graph.ensureGpuBuffer(Tensor.ones(1, 4));
        try (var scope = BackwardBuffers.begin()) {
            aliases = scope.capture(() -> chain(input).stream().map(Tensor::aliasStorage).toList());
        }
        assertEquals(4, graph.backwardBufferIds.size());
        assertEquals(3, GraphBufferReuse.prepare(graph).allocations().size());
        graph.flush();
        assertEquals(1, graph.allocatedBufferIds.size());
        assertTrue(aliases.stream().allMatch(value -> value.getGpuTag() == null));
    }

    private List<Tensor> chain(GpuBuffer input) {
        var values = new java.util.ArrayList<Tensor>();
        for (int index = 0; index < 4; index++) {
            Tensor value = output();
            graph.recordUnary(ComputeGraph.OP_NEG, input, (GpuBuffer) value.getGpuTag());
            input = (GpuBuffer) value.getGpuTag();
            values.add(value);
        }
        return values;
    }
}
