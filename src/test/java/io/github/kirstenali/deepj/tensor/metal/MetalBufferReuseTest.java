package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.tensor.ComputeGraph;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MetalBufferReuseTest extends MetalDifferentialTestSupport {
    @Test void recycledStorageHonorsNewDtypeUploadsAndZeroInitialization() {
        var graph = new ComputeGraph(new MetalGpuRuntime());
        Tensor retained = Tensor.from2D(new float[][]{{3, 5}});
        int keeper = graph.ensureGpuBuffer(retained).id;
        graph.flush();
        try {
            for (int iteration = 0; iteration < 4; iteration++) { recycle(900000001 + iteration); }
            float[] actual = new float[2];
            MetalNative.nativeDownloadBuffer(keeper, actual);
            assertArrayEquals(new float[]{3, 5}, actual);
        } finally { graph.releaseAll(); }
    }

    private static void recycle(int id) {
        int[] ids = {id};
        try {
            MetalNative.nativeAllocBuffers(ids, new int[]{64}, 1);
            MetalNative.nativeUploadBuffer(id, new float[64]);
            MetalNative.nativeReleaseBuffers(ids, 1);
            MetalNative.nativeAllocTypedBuffers(ids, new int[]{128}, new int[]{3}, 1);
            short[] values = new short[128];
            java.util.Arrays.fill(values, (short) 0x3f80);
            MetalNative.nativeUploadBuffer16(id, values);
            MetalNative.nativeZeroTypedBuffer(id, 128, 3);
            MetalNative.nativeDownloadBuffer16(id, values);
            assertArrayEquals(new short[128], values);
            assertEquals(256, MetalNative.nativeBufferByteLength(id));
        } finally { MetalNative.nativeReleaseBuffers(ids, 1); }
    }

    @Test void explicitCacheReleaseLeavesLiveBuffersIntact() {
        var graph = new ComputeGraph(new MetalGpuRuntime());
        int id = graph.ensureGpuBuffer(Tensor.from2D(new float[][]{{7}})).id;
        graph.flush();
        try {
            MetalNative.nativeClearBufferCache();
            float[] values = new float[1];
            MetalNative.nativeDownloadBuffer(id, values);
            assertArrayEquals(new float[]{7}, values);
        } finally { graph.releaseAll(); }
    }
}
