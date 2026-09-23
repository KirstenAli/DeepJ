package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ComputeGraphTest extends ComputeGraphTestSupport {

    @Test
    void constructorRejectsNull() {
        assertThrows(NullPointerException.class, () -> new ComputeGraph(null));
    }

    @Test
    void newGraphIsEmpty() {
        assertTrue(graph.isEmpty());
    }

    @Test
    void ensureGpuBufferAssignsGpuTag() {
        Tensor t = Tensor.from2D(new float[][]{{1, 2}, {3, 4}});
        assertNull(t.getGpuTag());

        GpuBuffer buf = graph.ensureGpuBuffer(t);

        assertNotNull(buf);
        assertSame(buf, t.getGpuTag());
        assertEquals(2, buf.rows);
        assertEquals(2, buf.cols);
    }

    @Test
    void ensureGpuBufferReusesExistingBuffer() {
        Tensor t = new Tensor(2, 3);
        GpuBuffer first = graph.ensureGpuBuffer(t);
        GpuBuffer second = graph.ensureGpuBuffer(t);

        assertSame(first, second, "should reuse existing GpuBuffer");
        graph.flush();
        assertEquals(1, runtime.uploads.size());
    }

    @Test
    void ensureGpuBufferReUploadsWhenNeedsUpload() {
        Tensor t = Tensor.from2D(new float[][]{{1.0f}});
        GpuBuffer buf = graph.ensureGpuBuffer(t);

        buf.needsUpload = true;
        t.data[0] = 99.0f;

        graph.ensureGpuBuffer(t);
        assertFalse(buf.needsUpload, "needsUpload should be cleared after re-scheduling");

        graph.flush();

        assertEquals(2, runtime.uploads.size());
    }

    @Test
    void newOutputBufferIsCpuStale() {
        GpuBuffer buf = graph.newOutputBuffer(4, 8);

        assertTrue(buf.cpuStale);
        assertFalse(buf.allocatedOnGpu);
        assertEquals(32, buf.floatCount());
    }

    @Test
    void outputBufferIdsAreIncreasing() {
        GpuBuffer a = graph.newOutputBuffer(1, 1);
        GpuBuffer b = graph.newOutputBuffer(1, 1);

        assertTrue(b.id > a.id, "buffer ids should be monotonically increasing");
    }

    @Test
    void createOutputTensorLinksBufferToTensor() {
        GpuBuffer buf = graph.newOutputBuffer(3, 5);
        Tensor t = graph.createOutputTensor(buf);

        assertEquals(3, t.rows);
        assertEquals(5, t.cols);
        assertSame(buf, t.getGpuTag());
    }

    @Test
    void recordBinaryMakesGraphNonEmpty() {
        GpuBuffer a = graph.newOutputBuffer(2, 2);
        GpuBuffer b = graph.newOutputBuffer(2, 2);
        GpuBuffer out = graph.newOutputBuffer(2, 2);

        assertTrue(graph.isEmpty());
        graph.recordBinary(ComputeGraph.OP_ADD, a, b, out);
        assertFalse(graph.isEmpty());
    }

    @Test
    void recordUnaryMakesGraphNonEmpty() {
        GpuBuffer in = graph.newOutputBuffer(2, 2);
        GpuBuffer out = graph.newOutputBuffer(2, 2);

        graph.recordUnary(ComputeGraph.OP_SQRT, in, out);
        assertFalse(graph.isEmpty());
    }

    @Test
    void recordSumSquaresMakesGraphNonEmpty() {
        GpuBuffer input = graph.newOutputBuffer(2, 3);
        GpuBuffer output = graph.newOutputBuffer(2, 1);
        graph.recordSumSquares(input, output, 2, 3);
        assertFalse(graph.isEmpty());
    }

    @Test
    void recordMatmulMakesGraphNonEmpty() {
        GpuBuffer a = graph.newOutputBuffer(2, 3);
        GpuBuffer b = graph.newOutputBuffer(3, 4);
        GpuBuffer out = graph.newOutputBuffer(2, 4);

        graph.recordMatmul(a, b, out, 2, 4, 3);
        assertFalse(graph.isEmpty());
    }

    @Test
    void recordAttentionOperationsMakeGraphNonEmpty() {
        GpuBuffer input = graph.newOutputBuffer(6, 4);
        GpuBuffer output = graph.newOutputBuffer(6, 4);
        graph.recordHeadPermutation(ComputeGraph.OP_SPLIT_HEADS,
                input, output, 3, 2, 2, 4);
        assertFalse(graph.isEmpty());
    }

    @Test
    void recordMultiplyScalarMakesGraphNonEmpty() {
        GpuBuffer in = graph.newOutputBuffer(2, 2);
        GpuBuffer out = graph.newOutputBuffer(2, 2);

        graph.recordMultiplyScalar(in, out, 2.5f);
        assertFalse(graph.isEmpty());
    }

    @Test
    void recordSoftmaxRowsMakesGraphNonEmpty() {
        GpuBuffer in = graph.newOutputBuffer(4, 8);
        GpuBuffer out = graph.newOutputBuffer(4, 8);

        graph.recordSoftmaxRows(in, out, 4, 8);
        assertFalse(graph.isEmpty());
    }

    @Test
    void recordSoftmaxBackwardMakesGraphNonEmpty() {
        GpuBuffer grad = graph.newOutputBuffer(4, 8);
        GpuBuffer probs = graph.newOutputBuffer(4, 8);
        GpuBuffer out = graph.newOutputBuffer(4, 8);

        graph.recordSoftmaxBackward(grad, probs, out, 4, 8);
        assertFalse(graph.isEmpty());
    }

    @Test
    void recordLayerNormBackwardMakesGraphNonEmpty() {
        GpuBuffer dXHat = graph.newOutputBuffer(4, 8);
        GpuBuffer xHat = graph.newOutputBuffer(4, 8);
        GpuBuffer std = graph.newOutputBuffer(4, 1);
        GpuBuffer out = graph.newOutputBuffer(4, 8);

        graph.recordLayerNormBackward(dXHat, xHat, std, out, 4, 8);
        assertFalse(graph.isEmpty());
    }

    @Test
    void flushOnEmptyGraphIsNoOp() {
        graph.flush();

        assertTrue(runtime.allocCalls.isEmpty());
        assertTrue(runtime.uploads.isEmpty());
        assertTrue(runtime.flushCalls.isEmpty());
    }

    @Test
    void flushAllocatesUploadsAndExecutes() {
        Tensor t = Tensor.from2D(new float[][]{{1, 2}, {3, 4}});
        GpuBuffer in = graph.ensureGpuBuffer(t);
        GpuBuffer out = graph.newOutputBuffer(2, 2);
        graph.recordUnary(ComputeGraph.OP_NEG, in, out);

        graph.flush();

        assertEquals(1, runtime.allocCalls.size());
        int[] allocIds = runtime.allocCalls.get(0).ids();
        assertEquals(2, allocIds.length);

        assertEquals(1, runtime.uploads.size());
        assertEquals(in.id, runtime.uploads.get(0).bufId());

        assertEquals(1, runtime.flushCalls.size());
        assertTrue(runtime.flushCalls.get(0).cmdStreamLength() > 0);
    }

    @Test
    void flushResetsOpStream() {
        GpuBuffer in = graph.newOutputBuffer(1, 1);
        GpuBuffer out = graph.newOutputBuffer(1, 1);
        graph.recordUnary(ComputeGraph.OP_EXP, in, out);

        assertFalse(graph.isEmpty());
        graph.flush();
        assertTrue(graph.isEmpty(), "graph should be empty after flush");
    }

    @Test
    void multipleFlushesAreIdempotent() {
        graph.flush();
        graph.flush();
        graph.flush();

        assertTrue(runtime.allocCalls.isEmpty());
        assertTrue(runtime.flushCalls.isEmpty());
    }

    @Test
    void secondFlushOnlyExecutesNewOps() {
        Tensor t = Tensor.from2D(new float[][]{{1}});
        GpuBuffer in = graph.ensureGpuBuffer(t);
        GpuBuffer out1 = graph.newOutputBuffer(1, 1);
        graph.recordUnary(ComputeGraph.OP_SQRT, in, out1);
        graph.flush();

        int allocCallsAfterFirst = runtime.allocCalls.size();
        int flushCallsAfterFirst = runtime.flushCalls.size();

        GpuBuffer out2 = graph.newOutputBuffer(1, 1);
        graph.recordUnary(ComputeGraph.OP_NEG, in, out2);
        graph.flush();

        assertEquals(allocCallsAfterFirst + 1, runtime.allocCalls.size(), "should allocate the new output buffer");
        assertEquals(flushCallsAfterFirst + 1, runtime.flushCalls.size(), "should execute the new op");
    }

    @Test
    void cmdStreamGrowsForManyOps() {

        GpuBuffer a = graph.newOutputBuffer(2, 2);
        GpuBuffer b = graph.newOutputBuffer(2, 2);

        for (int i = 0; i < 1000; i++) {
            GpuBuffer out = graph.newOutputBuffer(2, 2);
            graph.recordBinary(ComputeGraph.OP_ADD, a, b, out);
        }

        assertDoesNotThrow(() -> graph.flush());
        assertEquals(1, runtime.flushCalls.size());
    }


}
