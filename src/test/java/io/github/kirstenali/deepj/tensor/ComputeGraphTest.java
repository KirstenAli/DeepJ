package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ComputeGraphTest {

    private RecordingRuntime runtime;
    private ComputeGraph graph;

    @BeforeEach
    void setUp() {
        runtime = new RecordingRuntime();
        graph = new ComputeGraph(runtime);
    }

    // ── constructor ────────────────────────────────────────────────

    @Test
    void constructorRejectsNull() {
        assertThrows(NullPointerException.class, () -> new ComputeGraph(null));
    }

    @Test
    void newGraphIsEmpty() {
        assertTrue(graph.isEmpty());
    }

    // ── ensureGpuBuffer ────────────────────────────────────────────

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
    }

    @Test
    void ensureGpuBufferReUploadsWhenNeedsUpload() {
        Tensor t = Tensor.from2D(new float[][]{{1.0f}});
        GpuBuffer buf = graph.ensureGpuBuffer(t);

        // Simulate CPU modification (e.g. after adamWUpdate)
        buf.needsUpload = true;
        t.data[0] = 99.0f;

        graph.ensureGpuBuffer(t);
        assertFalse(buf.needsUpload, "needsUpload should be cleared after re-scheduling");

        // Flush and verify the re-upload happened
        graph.flush();
        // First upload from ensureGpuBuffer + second re-upload = 2 uploads
        assertEquals(2, runtime.uploads.size());
    }

    // ── newOutputBuffer ────────────────────────────────────────────

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

    // ── createOutputTensor ─────────────────────────────────────────

    @Test
    void createOutputTensorLinksBufferToTensor() {
        GpuBuffer buf = graph.newOutputBuffer(3, 5);
        Tensor t = graph.createOutputTensor(buf);

        assertEquals(3, t.rows);
        assertEquals(5, t.cols);
        assertSame(buf, t.getGpuTag());
    }

    // ── op recording ───────────────────────────────────────────────

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
    void recordMatmulMakesGraphNonEmpty() {
        GpuBuffer a = graph.newOutputBuffer(2, 3);
        GpuBuffer b = graph.newOutputBuffer(3, 4);
        GpuBuffer out = graph.newOutputBuffer(2, 4);

        graph.recordMatmul(a, b, out, 2, 4, 3);
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

    // ── flush pipeline ─────────────────────────────────────────────

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

        // Allocations: input buffer + output buffer
        assertEquals(1, runtime.allocCalls.size());
        int[] allocIds = runtime.allocCalls.get(0).ids;
        assertEquals(2, allocIds.length);

        // Upload: input tensor data
        assertEquals(1, runtime.uploads.size());
        assertEquals(in.id, runtime.uploads.get(0).bufId);

        // Execution: one flushOps call
        assertEquals(1, runtime.flushCalls.size());
        assertTrue(runtime.flushCalls.get(0).cmdStreamLength > 0);
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

        // Second round: new op on existing buffers
        GpuBuffer out2 = graph.newOutputBuffer(1, 1);
        graph.recordUnary(ComputeGraph.OP_NEG, in, out2);
        graph.flush();

        assertEquals(allocCallsAfterFirst + 1, runtime.allocCalls.size(), "should allocate the new output buffer");
        assertEquals(flushCallsAfterFirst + 1, runtime.flushCalls.size(), "should execute the new op");
    }

    // ── cmd stream growth ──────────────────────────────────────────

    @Test
    void cmdStreamGrowsForManyOps() {
        // Record more ops than the initial 4096 ints can hold
        // Each binary op uses 5 ints, so 1000 ops = 5000 ints > 4096
        GpuBuffer a = graph.newOutputBuffer(2, 2);
        GpuBuffer b = graph.newOutputBuffer(2, 2);

        for (int i = 0; i < 1000; i++) {
            GpuBuffer out = graph.newOutputBuffer(2, 2);
            graph.recordBinary(ComputeGraph.OP_ADD, a, b, out);
        }

        // Should not throw — cmd stream should have grown
        assertDoesNotThrow(() -> graph.flush());
        assertEquals(1, runtime.flushCalls.size());
    }

    // ── materialize ────────────────────────────────────────────────

    @Test
    void materializeFlushesAndDownloads() {
        Tensor input = Tensor.from2D(new float[][]{{2.0f, 4.0f}});
        GpuBuffer in = graph.ensureGpuBuffer(input);
        GpuBuffer outBuf = graph.newOutputBuffer(1, 2);
        graph.recordUnary(ComputeGraph.OP_SQRT, in, outBuf);

        Tensor result = graph.createOutputTensor(outBuf);
        assertTrue(outBuf.cpuStale);

        // Pre-load the download result the runtime will return
        runtime.downloadResult = new float[]{1.414f, 2.0f};

        graph.materialize(result);

        // Should have flushed
        assertEquals(1, runtime.flushCalls.size());
        // Should have downloaded
        assertEquals(1, runtime.downloads.size());
        assertEquals(outBuf.id, runtime.downloads.get(0).bufId);
        // CPU data should be updated
        assertEquals(1.414f, result.data[0], 1e-3f);
        assertEquals(2.0f, result.data[1], 1e-3f);
        // No longer stale
        assertFalse(outBuf.cpuStale);
    }

    @Test
    void materializeSkipsNonGpuTensor() {
        Tensor cpuOnly = new Tensor(2, 2);
        // Should be a no-op, no exceptions
        graph.materialize(cpuOnly);
        assertTrue(runtime.downloads.isEmpty());
    }

    @Test
    void materializeSkipsAlreadyFreshBuffer() {
        Tensor t = Tensor.from2D(new float[][]{{1}});
        GpuBuffer buf = graph.ensureGpuBuffer(t);
        buf.cpuStale = false;

        graph.materialize(t);
        assertTrue(runtime.downloads.isEmpty(), "should not download if not stale");
    }

    @Test
    void doubleMaterializeDoesNotDownloadTwice() {
        Tensor input = Tensor.from2D(new float[][]{{5.0f}});
        GpuBuffer in = graph.ensureGpuBuffer(input);
        GpuBuffer outBuf = graph.newOutputBuffer(1, 1);
        graph.recordUnary(ComputeGraph.OP_NEG, in, outBuf);

        Tensor result = graph.createOutputTensor(outBuf);
        runtime.downloadResult = new float[]{-5.0f};

        graph.materialize(result);
        graph.materialize(result);

        assertEquals(1, runtime.downloads.size(), "second materialize should be a no-op");
    }

    // ── releaseAll ─────────────────────────────────────────────────

    @Test
    void releaseAllReleasesBuffersAndResetsState() {
        Tensor t = Tensor.from2D(new float[][]{{1, 2}});
        graph.ensureGpuBuffer(t);
        GpuBuffer out = graph.newOutputBuffer(1, 2);
        graph.recordUnary(ComputeGraph.OP_NEG, graph.ensureGpuBuffer(t), out);

        graph.releaseAll();

        assertEquals(1, runtime.releaseCalls.size());
        assertTrue(graph.isEmpty(), "graph should be empty after releaseAll");

        // Should be able to start fresh
        GpuBuffer fresh = graph.ensureGpuBuffer(Tensor.from2D(new float[][]{{9}}));
        assertEquals(0, fresh.id, "buffer ids should restart from 0 after releaseAll");
    }

    @Test
    void releaseAllOnEmptyGraphIsNoOp() {
        graph.releaseAll();
        assertTrue(runtime.releaseCalls.isEmpty());
    }

    // ── op codes are distinct ──────────────────────────────────────

    @Test
    void allOpCodesAreUnique() {
        int[] codes = {
                ComputeGraph.OP_ADD, ComputeGraph.OP_SUBTRACT,
                ComputeGraph.OP_MULTIPLY, ComputeGraph.OP_DIVIDE,
                ComputeGraph.OP_MATMUL, ComputeGraph.OP_MULTIPLY_SCALAR,
                ComputeGraph.OP_SQRT, ComputeGraph.OP_NEG,
                ComputeGraph.OP_EXP, ComputeGraph.OP_LOG,
                ComputeGraph.OP_TANH, ComputeGraph.OP_SIGMOID,
                ComputeGraph.OP_RELU, ComputeGraph.OP_RELU_BACKWARD,
                ComputeGraph.OP_GELU, ComputeGraph.OP_GELU_BACKWARD,
                ComputeGraph.OP_SOFTMAX_ROWS, ComputeGraph.OP_SOFTMAX_BACKWARD,
                ComputeGraph.OP_LAYERNORM_BACKWARD, ComputeGraph.OP_ADAMW_UPDATE,
                ComputeGraph.OP_ADD_SCALAR, ComputeGraph.OP_DIVIDE_SCALAR,
                ComputeGraph.OP_TRANSPOSE, ComputeGraph.OP_ADD_ROW_VECTOR,
                ComputeGraph.OP_ADD_BROADCAST_COLS, ComputeGraph.OP_SUBTRACT_BROADCAST_COLS,
                ComputeGraph.OP_DIVIDE_BROADCAST_COLS, ComputeGraph.OP_MULTIPLY_BROADCAST_ROWS,
                ComputeGraph.OP_SUM_ROWS, ComputeGraph.OP_MEAN_ALONG_ROWS,
                ComputeGraph.OP_VARIANCE_ALONG_ROWS, ComputeGraph.OP_MULTIPLY_BROADCAST_COLS,
                ComputeGraph.OP_SUM_ALONG_ROWS, ComputeGraph.OP_MAX_ALONG_ROWS,
                ComputeGraph.OP_CLAMP, ComputeGraph.OP_POW,
                ComputeGraph.OP_SCATTER_ADD_ROWS, ComputeGraph.OP_SUM_ABS,
                ComputeGraph.OP_CROSS_ENTROPY_LOSS
        };
        assertEquals(39, codes.length);
        assertEquals(codes.length, java.util.Arrays.stream(codes).distinct().count(),
                "all op codes must be unique");
    }

    @Test
    void releaseAllMaterializesStaleTrackedTensor() {
        Tensor t = Tensor.from2D(new float[][]{{1.0f}});
        GpuBuffer buf = graph.ensureGpuBuffer(t);
        graph.flush();

        // Simulate a GPU-only in-place update before release.
        buf.cpuStale = true;
        runtime.downloadResult = new float[]{42.0f};

        graph.releaseAll();

        assertEquals(42.0f, t.data[0], 1e-6f, "releaseAll should preserve latest GPU value");
        assertNull(t.getGpuTag(), "releaseAll should still clear GPU tags");
        assertFalse(buf.cpuStale, "buffer state should be marked fresh after forced materialization");
    }

    // ── recordAdamWUpdate ──────────────────────────────────────────

    @Test
    void recordAdamWUpdateMakesGraphNonEmpty() {
        GpuBuffer w  = graph.newOutputBuffer(4, 4);
        GpuBuffer g  = graph.newOutputBuffer(4, 4);
        GpuBuffer mt = graph.newOutputBuffer(4, 4);
        GpuBuffer vt = graph.newOutputBuffer(4, 4);

        graph.recordAdamWUpdate(w, g, mt, vt, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f, 16);

        assertFalse(graph.isEmpty());
    }

    @Test
    void recordAdamWUpdateEncodesThirteenInts() {
        GpuBuffer w  = graph.newOutputBuffer(1, 4);
        GpuBuffer g  = graph.newOutputBuffer(1, 4);
        GpuBuffer mt = graph.newOutputBuffer(1, 4);
        GpuBuffer vt = graph.newOutputBuffer(1, 4);

        graph.recordAdamWUpdate(w, g, mt, vt, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.9f, 0.999f, 4);
        graph.flush();

        assertEquals(1, runtime.flushCalls.size());
        int len    = runtime.flushCalls.get(0).cmdStreamLength();
        int[] stream = runtime.flushCalls.get(0).cmdStream();
        assertEquals(13, len, "AdamW op must encode exactly 13 ints");
        assertEquals(ComputeGraph.OP_ADAMW_UPDATE,         stream[0]);
        assertEquals(w.id,                                 stream[1]);
        assertEquals(g.id,                                 stream[2]);
        assertEquals(mt.id,                                stream[3]);
        assertEquals(vt.id,                                stream[4]);
        assertEquals(Float.floatToRawIntBits(1e-3f),       stream[5]);
        assertEquals(Float.floatToRawIntBits(0.9f),        stream[6]);
        assertEquals(Float.floatToRawIntBits(0.999f),      stream[7]);
        assertEquals(Float.floatToRawIntBits(1e-8f),       stream[8]);
        assertEquals(Float.floatToRawIntBits(0.01f),       stream[9]);
        assertEquals(Float.floatToRawIntBits(0.9f),        stream[10]);
        assertEquals(Float.floatToRawIntBits(0.999f),      stream[11]);
        assertEquals(4,                                    stream[12]);
    }

    // ── markAllocatedBuffers ───────────────────────────────────────

    @Test
    void flushMarksBuffersAsAllocatedOnGpu() {
        Tensor t      = Tensor.from2D(new float[][]{{1.0f}});
        GpuBuffer in  = graph.ensureGpuBuffer(t);
        GpuBuffer out = graph.newOutputBuffer(1, 1);
        // createOutputTensor registers 'out' in the tracking map so markAllocatedBuffers can reach it
        graph.createOutputTensor(out);

        assertFalse(in.allocatedOnGpu,  "should not be allocated before flush");
        assertFalse(out.allocatedOnGpu, "should not be allocated before flush");

        graph.recordUnary(ComputeGraph.OP_NEG, in, out);
        graph.flush();

        assertTrue(in.allocatedOnGpu,  "input buffer should be marked allocated after flush");
        assertTrue(out.allocatedOnGpu, "output buffer should be marked allocated after flush");
    }

    // ── flush edge cases ───────────────────────────────────────────

    @Test
    void flushWithPendingAllocsButNoOpsAllocatesWithoutExecuting() {
        // newOutputBuffer queues an alloc but we record no ops
        graph.newOutputBuffer(2, 4);
        assertTrue(graph.isEmpty(), "no ops recorded");

        graph.flush();

        assertEquals(1, runtime.allocCalls.size(),  "pending alloc should have been submitted");
        assertTrue(runtime.flushCalls.isEmpty(),     "no ops → no flushOps call");
    }

    // ── orphan buffer detection ────────────────────────────────────

    @Test
    void bufferWhoseTagWasReplacedIsReleasedOnFlush() {
        Tensor t      = Tensor.from2D(new float[][]{{1.0f}});
        GpuBuffer old = graph.ensureGpuBuffer(t);
        graph.flush(); // allocate and upload

        // Attach tensor to a brand-new buffer — old becomes orphaned
        GpuBuffer replacement = graph.newOutputBuffer(1, 1);
        t.setGpuTag(replacement);

        int releasesBefore = runtime.releaseCalls.size();
        graph.flush();

        assertEquals(releasesBefore + 1, runtime.releaseCalls.size(),
                "one release call should follow the flush");
        int oldId     = old.id;
        boolean found = java.util.Arrays.stream(
                        runtime.releaseCalls.get(runtime.releaseCalls.size() - 1).ids())
                .anyMatch(id -> id == oldId);
        assertTrue(found, "the orphaned buffer id must appear in the release call");
    }

    @Test
    void bufferWithClearedGpuTagIsReleasedOnFlush() {
        Tensor t = Tensor.from2D(new float[][]{{2.0f}});
        GpuBuffer buf = graph.ensureGpuBuffer(t);
        graph.flush();

        t.setGpuTag(null); // tensor disowned the buffer
        int releasesBefore = runtime.releaseCalls.size();
        graph.flush();

        assertEquals(releasesBefore + 1, runtime.releaseCalls.size(),
                "buffer with no owning tensor must be released");
        int bufId     = buf.id;
        boolean found = java.util.Arrays.stream(
                        runtime.releaseCalls.get(runtime.releaseCalls.size() - 1).ids())
                .anyMatch(id -> id == bufId);
        assertTrue(found, "the released id must match the detached buffer");
    }

    // ── releaseAll clears tensor GPU tags ──────────────────────────

    @Test
    void releaseAllClearsTensorGpuTags() {
        Tensor t = Tensor.from2D(new float[][]{{3.0f}});
        graph.ensureGpuBuffer(t);

        assertNotNull(t.getGpuTag(), "gpu tag should be set before releaseAll");
        graph.releaseAll();
        assertNull(t.getGpuTag(), "gpu tag should be null after releaseAll");
    }

    // ═══════════════════════════════════════════════════════════════
    //  Recording GpuRuntime stub
    // ═══════════════════════════════════════════════════════════════

    /** Simple stub that records every call for verification. */
    static class RecordingRuntime implements GpuRuntime {

        record AllocCall(int[] ids, int[] sizes, int count) {}
        record UploadCall(int bufId, float[] data) {}
        record DownloadCall(int bufId) {}
        record FlushCall(int[] cmdStream, int cmdStreamLength) {}
        record ReleaseCall(int[] ids, int count) {}

        final List<AllocCall> allocCalls = new ArrayList<>();
        final List<UploadCall> uploads = new ArrayList<>();
        final List<DownloadCall> downloads = new ArrayList<>();
        final List<FlushCall> flushCalls = new ArrayList<>();
        final List<ReleaseCall> releaseCalls = new ArrayList<>();

        /** Data to copy into the out[] array on the next downloadBuffer call. */
        float[] downloadResult = null;

        @Override
        public void allocBuffers(int[] ids, int[] sizes, int count) {
            allocCalls.add(new AllocCall(ids.clone(), sizes.clone(), count));
        }

        @Override
        public void uploadBuffer(int bufId, float[] data) {
            uploads.add(new UploadCall(bufId, data.clone()));
        }

        @Override
        public void downloadBuffer(int bufId, float[] out) {
            downloads.add(new DownloadCall(bufId));
            if (downloadResult != null) {
                System.arraycopy(downloadResult, 0, out, 0,
                        Math.min(downloadResult.length, out.length));
            }
        }

        @Override
        public void releaseBuffers(int[] ids, int count) {
            releaseCalls.add(new ReleaseCall(ids.clone(), count));
        }

        @Override
        public void flushOps(int[] cmdStream, int cmdStreamLength) {
            flushCalls.add(new FlushCall(cmdStream.clone(), cmdStreamLength));
        }
    }
}

