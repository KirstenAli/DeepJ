package io.github.kirstenali.deepj.tensor;

/** Returns managed ID leases only after a successful native release. */
final class GraphBufferRelease {
    private GraphBufferRelease() {}

    static void release(GraphCore graph, int[] ids) {
        graph.runtime.releaseBuffers(ids, ids.length);
        int[] recycled = new int[ids.length];
        int count = 0;
        for (int id : ids) {
            GpuBuffer buffer = graph.managedBuffers.remove(id);
            if (buffer != null) {
                buffer.markReleased();
                recycled[count++] = id;
            }
            graph.allocatedBufferIds.remove(id);
            graph.backwardBufferIds.remove(id);
        }
        GpuBufferIds.SHARED.release(recycled, count);
    }
}
