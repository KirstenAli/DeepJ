package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.Test;

import java.lang.ref.WeakReference;

import static org.junit.jupiter.api.Assertions.*;

class GpuScratchDTypeTest extends ComputeGraphTestSupport {
    @Test
    void scratchReuseRequiresIdenticalFormatNotJustIdenticalBytesOrElementCount() {
        GpuBuffer input = graph.newOutputBuffer(1, 4), output = graph.newOutputBuffer(1, 4);
        GpuBuffer half = scratch(DType.FLOAT16), bfloat = scratch(DType.BFLOAT16), nextHalf = scratch(DType.FLOAT16);
        int[] lifetimes = {ComputeGraph.OP_NEG, input.id, half.id, 4,
                ComputeGraph.OP_NEG, half.id, output.id, 4, ComputeGraph.OP_NEG, input.id, bfloat.id, 4,
                ComputeGraph.OP_NEG, bfloat.id, output.id, 4, ComputeGraph.OP_NEG, input.id, nextHalf.id, 4,
                ComputeGraph.OP_NEG, nextHalf.id, output.id, 4};
        graph.cmdStream = lifetimes;
        graph.cmdPos = lifetimes.length;
        // This is a lifetime-planner fixture, not executable typed matrix arithmetic.
        var plan = GraphBufferReuse.prepare(graph);
        assertEquals(4, plan.allocations().size());
        assertEquals(half.id, plan.commands()[18]);
        assertTrue(plan.allocations().stream().anyMatch(allocation -> allocation[0] == bfloat.id));
        assertFalse(plan.allocations().stream().anyMatch(allocation -> allocation[0] == nextHalf.id));
        assertArrayEquals(lifetimes, graph.cmdStream);
    }

    private GpuBuffer scratch(DType dtype) {
        GpuBuffer buffer = graph.newOutputBuffer(1, 4, dtype);
        graph.bufIdToMemory.put(buffer.id, new WeakReference<>(null));
        return buffer;
    }
}
