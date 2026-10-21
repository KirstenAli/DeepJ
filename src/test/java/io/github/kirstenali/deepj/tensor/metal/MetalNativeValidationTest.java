package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class MetalNativeValidationTest extends MetalDifferentialTestSupport {

    private ComputeGraph graph;
    private int buffer;

    @BeforeEach
    void allocateTransferBuffer() {
        graph = new ComputeGraph(new MetalGpuRuntime());
        buffer = graph.ensureGpuBuffer(Tensor.from2D(new float[][]{{1, 2}})).id;
        graph.flush();
    }

    @AfterEach
    void releaseTransferBuffer() {
        if (graph != null) graph.releaseAll();
    }

    @Test
    void allocationsRejectInvalidArraysCountsIdsAndSizes() {
        assertThrows(RuntimeException.class, () -> MetalNative.nativeAllocBuffers(null, new int[]{2}, 1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeAllocBuffers(new int[]{buffer}, null, 1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeAllocBuffers(new int[]{buffer}, new int[0], 1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeAllocBuffers(new int[0], new int[]{2}, 1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeAllocBuffers(new int[0], new int[0], -1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeAllocBuffers(new int[]{-1}, new int[]{2}, 1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeAllocBuffers(new int[]{buffer}, new int[]{0}, 1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeAllocBuffers(new int[]{buffer}, new int[]{3}, 1));
    }

    @Test
    void sameSizeReallocationPreservesBufferContents() {
        MetalNative.nativeAllocBuffers(new int[]{buffer}, new int[]{2}, 1);
        float[] values = new float[2];
        MetalNative.nativeDownloadBuffer(buffer, values);
        assertArrayEquals(new float[]{1, 2}, values);
    }

    @Test
    void transfersRejectMissingBuffersNullArraysAndWrongLengths() {
        assertThrows(RuntimeException.class, () -> MetalNative.nativeUploadBuffer(buffer, null));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeDownloadBuffer(buffer, null));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeUploadBuffer(-1, new float[2]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeDownloadBuffer(-1, new float[2]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeUploadBuffer(buffer, new float[1]));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeDownloadBuffer(buffer, new float[1]));
    }

    @Test
    void releasesRejectInvalidRequests() {
        assertThrows(RuntimeException.class, () -> MetalNative.nativeReleaseBuffers(null, 0));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeReleaseBuffers(new int[0], -1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeReleaseBuffers(new int[0], 1));
        MetalNative.nativeReleaseBuffers(new int[0], 0);
    }

    @ParameterizedTest
    @MethodSource("invalidCommands")
    void malformedCommandsAreRejectedBeforeEncoding(int[] commands) {
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(commands,
                commands == null ? 0 : commands.length));
    }

    private static Stream<int[]> invalidCommands() {
        return Stream.of(null, new int[]{Integer.MAX_VALUE}, new int[]{ComputeGraph.OP_ADD});
    }

    @Test
    void commandLengthsMustMatchTheArray() {
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(new int[0], -1));
        assertThrows(RuntimeException.class, () -> MetalNative.nativeFlushOps(new int[0], 1));
        MetalNative.nativeFlushOps(new int[0], 0);
    }
}
