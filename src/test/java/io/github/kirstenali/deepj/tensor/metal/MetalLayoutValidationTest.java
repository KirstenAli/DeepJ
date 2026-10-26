package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalLayoutValidationTest extends MetalDifferentialTestSupport {
    private final List<Tensor> owners = new ArrayList<>();
    @Test
    void descriptorHeadersRejectMissingTruncatedNegativeAndOverflowingFields() {
        int[] ids = buffers();
        invalid(ids, null, 0);
        invalid(ids, new long[0], 0);
        invalid(ids, new long[]{1, 4, 4}, 0);
        invalid(ids, new long[]{1, 4, 4, 0}, 0);
        invalid(ids, new long[]{-1, 4, 4, 0}, 0);
        invalid(ids, new long[]{0, 1, 4, 0}, -1);
        invalid(ids, new long[]{0, 1, 4, 0}, Integer.MAX_VALUE);
        invalid(ids, new long[]{Long.MAX_VALUE, 4, 4, 0}, 0);
        invalid(ids, new long[]{1, -1, 4, 0, 4, 1}, 0);
        invalid(ids, new long[]{1, 4, 3, 0, 4, 1}, 0);
        invalid(ids, new long[]{1, 4, 4, 4, 4, 1}, 0);
    }

    @Test
    void invalidAxesCountsAndExtentsAreRejectedBeforeKernelDispatch() {
        int[] ids = buffers();
        invalid(ids, new long[]{1, 4, 4, 0, 0, 1}, 0);
        invalid(ids, new long[]{1, 4, 4, 0, -1, 1}, 0);
        invalid(ids, new long[]{1, 4, 4, 0, 4, -1}, 0);
        invalid(ids, new long[]{1, 4, 4, 0, 4, Long.MAX_VALUE}, 0);
        invalid(ids, new long[]{1, 3, 4, 0, 4, 1}, 0);
        invalid(ids, new long[]{2, 4, 4, 0, 4, 2, 1, 1}, 0);
        invalid(ids, new long[]{1, 4, 4, 1, 4, 1}, 0);
        invalid(ids, new long[]{0, 4, 4, 0}, 0);
        invalid(new int[]{ids[0], ids[0]}, new long[]{1, 4, 4, 0, 4, 1}, 0);
    }

    @Test
    void layoutArrayAndCommandArrayValidationIsIndependentOfShaderExecution() {
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushLayoutOps(null, 0, new long[0]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushLayoutOps(new int[0], -1, new long[0]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushLayoutOps(new int[0], 1, new long[0]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushLayoutOps(new int[0], 0, null));
        MetalNative.nativeFlushLayoutOps(new int[0], 0, new long[0]);
        int[] ids = buffers();
        int[] command = {ComputeGraph.OP_LAYOUT_TRANSFER, ids[0], ids[1], 0};
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(command, command.length));
    }

    private int[] buffers() {
        Tensor source = new Tensor(1, 4), output = new Tensor(1, 4);
        owners.add(source);
        owners.add(output);
        int inputId = metal.graph.ensureGpuBuffer(source).id, outputId = metal.graph.ensureGpuBuffer(output).id;
        metal.graph.flush();
        return new int[]{inputId, outputId};
    }

    private static void invalid(int[] ids, long[] layouts, int offset) {
        int[] commands = {ComputeGraph.OP_LAYOUT_TRANSFER, ids[0], ids[1], offset};
        assertThrows(RuntimeException.class,
                () -> MetalNative.nativeFlushLayoutOps(commands, commands.length, layouts));
    }
}
