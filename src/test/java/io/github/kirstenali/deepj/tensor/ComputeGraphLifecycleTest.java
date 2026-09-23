package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ComputeGraphLifecycleTest extends ComputeGraphTestSupport {
    @Test
    void materializeFlushesAndDownloads() {
        Tensor input = Tensor.from2D(new float[][]{{2.0f, 4.0f}});
        GpuBuffer in = graph.ensureGpuBuffer(input);
        GpuBuffer outBuf = graph.newOutputBuffer(1, 2);
        graph.recordUnary(ComputeGraph.OP_SQRT, in, outBuf);

        Tensor result = graph.createOutputTensor(outBuf);
        assertTrue(outBuf.cpuStale);

        runtime.downloadResult = new float[]{1.414f, 2.0f};

        graph.materialize(result);

        assertEquals(1, runtime.flushCalls.size());

        assertEquals(1, runtime.downloads.size());
        assertEquals(outBuf.id, runtime.downloads.get(0).bufId());

        assertEquals(1.414f, result.data[0], 1e-3f);
        assertEquals(2.0f, result.data[1], 1e-3f);

        assertFalse(outBuf.cpuStale);
    }

    @Test
    void materializeSkipsNonGpuTensor() {
        Tensor cpuOnly = new Tensor(2, 2);

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

    @Test
    void releaseAllReleasesBuffersAndResetsState() {
        Tensor t = Tensor.from2D(new float[][]{{1, 2}});
        graph.ensureGpuBuffer(t);
        GpuBuffer out = graph.newOutputBuffer(1, 2);
        graph.recordUnary(ComputeGraph.OP_NEG, graph.ensureGpuBuffer(t), out);

        graph.releaseAll();

        assertEquals(1, runtime.releaseCalls.size());
        assertTrue(graph.isEmpty(), "graph should be empty after releaseAll");

        GpuBuffer fresh = graph.ensureGpuBuffer(Tensor.from2D(new float[][]{{9}}));
        assertTrue(fresh.id > out.id, "buffer ids must stay globally unique after releaseAll");
    }

    @Test
    void releaseAllOnEmptyGraphIsNoOp() {
        graph.releaseAll();
        assertTrue(runtime.releaseCalls.isEmpty());
    }

    @Test
    void releaseAllIncludesUnboundOutputBuffers() {
        graph.newOutputBuffer(2, 2);
        graph.releaseAll();
        assertEquals(1, runtime.releaseCalls.size());
        assertEquals(1, runtime.releaseCalls.get(0).count());
    }

    @Test
    void allOpCodesAreUnique() throws IllegalAccessException {
        var fields = java.util.Arrays.stream(ComputeGraph.class.getFields())
                .filter(field -> field.getName().startsWith("OP_"))
                .toList();
        int[] codes = new int[fields.size()];
        for (int index = 0; index < fields.size(); index++) {
            codes[index] = fields.get(index).getInt(null);
        }
        assertEquals(56, fields.size());
        assertEquals(fields.size(), java.util.Arrays.stream(codes).distinct().count(),
                "all op codes must be unique");
    }

    @Test
    void releaseAllMaterializesStaleTrackedTensor() {
        Tensor t = Tensor.from2D(new float[][]{{1.0f}});
        GpuBuffer buf = graph.ensureGpuBuffer(t);
        graph.flush();

        buf.cpuStale = true;
        runtime.downloadResult = new float[]{42.0f};

        graph.releaseAll();

        assertEquals(42.0f, t.data[0], 1e-6f, "releaseAll should preserve latest GPU value");
        assertNull(t.getGpuTag(), "releaseAll should still clear GPU tags");
        assertFalse(buf.cpuStale, "buffer state should be marked fresh after forced materialization");
    }

    @Test
    void releaseAllDownloadsDirectlyIntoTensorStorage() {
        Tensor tensor = Tensor.from2D(new float[][]{{1.0f}});
        GpuBuffer buffer = graph.ensureGpuBuffer(tensor);
        graph.flush();
        buffer.cpuStale = true;

        graph.releaseAll();

        assertSame(tensor.data, runtime.downloads.get(0).output());
    }

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
        assertAdamBufferIds(stream, w, g, mt, vt);
        assertAdamValues(stream);
    }

    private static void assertAdamBufferIds(int[] stream, GpuBuffer w, GpuBuffer g,
                                            GpuBuffer mt, GpuBuffer vt) {
        assertEquals(ComputeGraph.OP_ADAMW_UPDATE,         stream[0]);
        assertEquals(w.id,                                 stream[1]);
        assertEquals(g.id,                                 stream[2]);
        assertEquals(mt.id,                                stream[3]);
        assertEquals(vt.id,                                stream[4]);
    }

    private static void assertAdamValues(int[] stream) {
        assertEquals(Float.floatToRawIntBits(1e-3f),       stream[5]);
        assertEquals(Float.floatToRawIntBits(0.9f),        stream[6]);
        assertEquals(Float.floatToRawIntBits(0.999f),      stream[7]);
        assertEquals(Float.floatToRawIntBits(1e-8f),       stream[8]);
        assertEquals(Float.floatToRawIntBits(0.01f),       stream[9]);
        assertEquals(Float.floatToRawIntBits(0.9f),        stream[10]);
        assertEquals(Float.floatToRawIntBits(0.999f),      stream[11]);
        assertEquals(4,                                    stream[12]);
    }

    @Test
    void flushMarksBuffersAsAllocatedOnGpu() {
        Tensor t      = Tensor.from2D(new float[][]{{1.0f}});
        GpuBuffer in  = graph.ensureGpuBuffer(t);
        GpuBuffer out = graph.newOutputBuffer(1, 1);

        graph.createOutputTensor(out);

        assertFalse(in.allocatedOnGpu,  "should not be allocated before flush");
        assertFalse(out.allocatedOnGpu, "should not be allocated before flush");

        graph.recordUnary(ComputeGraph.OP_NEG, in, out);
        graph.flush();

        assertTrue(in.allocatedOnGpu,  "input buffer should be marked allocated after flush");
        assertTrue(out.allocatedOnGpu, "output buffer should be marked allocated after flush");
    }

    @Test
    void flushWithPendingAllocsButNoOpsAllocatesWithoutExecuting() {

        graph.newOutputBuffer(2, 4);
        assertTrue(graph.isEmpty(), "no ops recorded");

        graph.flush();

        assertEquals(1, runtime.allocCalls.size(),  "pending alloc should have been submitted");
        assertTrue(runtime.flushCalls.isEmpty(),     "no ops → no flushOps call");
    }

    @Test
    void bufferWhoseTagWasReplacedIsReleasedOnFlush() {
        Tensor t      = Tensor.from2D(new float[][]{{1.0f}});
        GpuBuffer old = graph.ensureGpuBuffer(t);
        graph.flush();

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

        t.setGpuTag(null);
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

    @Test
    void releaseAllClearsTensorGpuTags() {
        Tensor t = Tensor.from2D(new float[][]{{3.0f}});
        graph.ensureGpuBuffer(t);

        assertNotNull(t.getGpuTag(), "gpu tag should be set before releaseAll");
        graph.releaseAll();
        assertNull(t.getGpuTag(), "gpu tag should be null after releaseAll");
    }

    @Test
    void temporaryReleaseKeepsRetainedBuffer() {
        Tensor retained = Tensor.from2D(new float[][]{{1.0f}}).retainDeviceBuffer();
        Tensor temporary = outputFrom(retained);
        Object retainedTag = retained.getGpuTag();
        Object temporaryTag = temporary.getGpuTag();
        graph.releaseTemporary();
        assertSame(retainedTag, retained.getGpuTag());
        assertNull(temporary.getGpuTag());
        assertReleased(temporaryTag);
        assertNotReleased(retainedTag);
        assertTrue(runtime.downloads.isEmpty());
    }

    private Tensor outputFrom(Tensor input) {
        GpuBuffer source = graph.ensureGpuBuffer(input);
        GpuBuffer output = graph.newOutputBuffer(input.rows, input.cols);
        graph.recordUnary(ComputeGraph.OP_NEG, source, output);
        return graph.createOutputTensor(output);
    }

    private void assertReleased(Object tag) {
        int id = ((GpuBuffer) tag).id;
        assertTrue(java.util.Arrays.stream(lastRelease().ids()).anyMatch(value -> value == id));
    }

    private void assertNotReleased(Object tag) {
        int id = ((GpuBuffer) tag).id;
        assertFalse(java.util.Arrays.stream(lastRelease().ids()).anyMatch(value -> value == id));
    }

    private RecordingRuntime.ReleaseCall lastRelease() {
        return runtime.releaseCalls.get(runtime.releaseCalls.size() - 1);
    }


}
