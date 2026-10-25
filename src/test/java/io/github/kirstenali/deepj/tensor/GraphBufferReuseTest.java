package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.lang.ref.Reference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GraphBufferReuseTest extends ComputeGraphTestSupport {
    private final List<Tensor> owners = new ArrayList<>();
    @Test
    void unreachableDisjointScratchLifetimesShareStorageWithoutChangingLogicalCommands() {
        List<GpuBuffer> buffers = chain(5, 4);
        int[] original = Arrays.copyOf(graph.cmdStream, graph.cmdPos);
        unreachable(buffers.subList(1, 5));
        GraphBufferReuse.Plan plan = GraphBufferReuse.prepare(graph);
        assertEquals(4, plan.allocations().size());
        assertArrayEquals(original, Arrays.copyOf(graph.cmdStream, graph.cmdPos));
        assertEquals(buffers.get(1).id, plan.commands()[10]);
        graph.flush();
        assertEquals(4, runtime.allocCalls.get(0).count());
        assertEquals(4, runtime.releaseCalls.get(0).count());
    }

    @Test
    void liveAndExplicitlyRetainedTensorsNeverBecomeScratch() {
        List<GpuBuffer> buffers = chain(4, 4);
        List<Tensor> live = new ArrayList<>();
        for (GpuBuffer buffer : buffers) {
            live.add(graph.bufIdToTensor.get(buffer.id).get());
        }
        live.get(1).retainDeviceBuffer();
        GraphBufferReuse.Plan plan = GraphBufferReuse.prepare(graph);
        assertEquals(5, plan.allocations().size());
        assertSame(graph.cmdStream, plan.commands());
        Reference.reachabilityFence(live);
        Reference.reachabilityFence(owners);
    }

    @Test
    void uploadedInputsAreNotAliasedEvenWhenTheirTensorIsUnreachable() {
        List<GpuBuffer> buffers = chain(2, 4);
        unreachable(buffers);
        GraphBufferReuse.Plan plan = GraphBufferReuse.prepare(graph);
        assertEquals(3, plan.allocations().size());
        assertTrue(plan.allocations().stream().anyMatch(allocation -> allocation[0] == buffers.get(0).id));
    }

    @Test
    void unboundPublicBufferHandlesAreNotAssumedToBeUnreachableTensors() {
        GpuBuffer input = graph.ensureGpuBuffer(Tensor.ones(1, 4));
        GpuBuffer first = graph.newOutputBuffer(1, 4), second = graph.newOutputBuffer(1, 4);
        graph.recordUnary(ComputeGraph.OP_NEG, input, first);
        graph.recordUnary(ComputeGraph.OP_NEG, first, second);
        assertEquals(3, GraphBufferReuse.prepare(graph).allocations().size());
    }

    @Test
    void overlappingLifetimesAndDifferentAllocationSizesRemainDistinct() {
        List<GpuBuffer> buffers = chain(3, 4);
        unreachable(buffers.subList(1, 3));
        assertEquals(4, GraphBufferReuse.prepare(graph).allocations().size());
        GpuBuffer large = graph.newOutputBuffer(1, 8);
        graph.createOutputTensor(large);
        graph.bufIdToTensor.get(large.id).clear();
        graph.recordUnary(ComputeGraph.OP_NEG, buffers.get(3), large);
        assertEquals(5, GraphBufferReuse.prepare(graph).allocations().size());
    }

    @Test
    void sharedStorageCanBeReusedAcrossMoreThanTwoLogicalBuffers() {
        List<GpuBuffer> buffers = chain(8, 4);
        unreachable(buffers.subList(1, 8));
        var plan = GraphBufferReuse.prepare(graph);
        assertEquals(4, plan.allocations().size());
        assertEquals(buffers.get(1).id, plan.commands()[26]);
    }

    @Test
    void unknownOrTruncatedCommandsAreRejectedBeforeAllocating() {
        graph.cmdStream[0] = -1;
        graph.cmdPos = 1;
        assertThrows(IllegalArgumentException.class, () -> GraphBufferReuse.prepare(graph));
        graph.cmdStream[0] = ComputeGraph.OP_NEG;
        assertThrows(IllegalArgumentException.class, () -> GraphBufferReuse.prepare(graph));
        graph.cmdStream[0] = ComputeGraph.OP_METADATA.length;
        assertThrows(IllegalArgumentException.class, () -> GraphBufferReuse.prepare(graph));
        assertTrue(runtime.allocCalls.isEmpty());
    }

    @Test
    void unusedPendingBuffersDoNotChangeCommandsOrBorrowAnotherLifetime() {
        graph.createOutputTensor(graph.newOutputBuffer(1, 4));
        graph.bufIdToTensor.values().forEach(Reference::clear);
        var plan = GraphBufferReuse.prepare(graph);
        assertEquals(1, plan.allocations().size());
        assertSame(graph.cmdStream, plan.commands());
    }

    private List<GpuBuffer> chain(int count, int width) {
        List<GpuBuffer> buffers = new ArrayList<>();
        buffers.add(graph.ensureGpuBuffer(Tensor.ones(1, width)));
        for (int index = 0; index < count; index++) {
            GpuBuffer output = graph.newOutputBuffer(1, width);
            owners.add(graph.createOutputTensor(output));
            graph.recordUnary(ComputeGraph.OP_NEG, buffers.get(buffers.size() - 1), output);
            buffers.add(output);
        }
        return buffers;
    }

    private void unreachable(List<GpuBuffer> buffers) {
        for (GpuBuffer buffer : buffers) {
            graph.bufIdToTensor.get(buffer.id).clear();
        }
    }
}
