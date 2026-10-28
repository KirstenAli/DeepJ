package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;

/** Uploads CPU copies directly to their destination, with bounded queued host snapshots. */
final class GraphHostCopies {
    static final long SNAPSHOT_LIMIT = 64L * 1024 * 1024;

    private GraphHostCopies() {}

    static Tensor copy(GraphExecution graph, Tensor source) {
        long bytes = source.dtype().storageBytes(source.numel());
        if (pendingBytes(graph) + bytes > SNAPSHOT_LIMIT) graph.flush();
        Object snapshot = snapshot(source);
        int rows = source.rank() == 2 ? source.matrixRows() : 1;
        GpuBuffer output = graph.newOutputBuffer(rows, source.numel() / rows, source.dtype());
        graph.scheduleUpload(output.id, snapshot);
        Tensor result = graph.createOutputLike(output, source);
        if (bytes > SNAPSHOT_LIMIT) graph.flush();
        return result;
    }

    private static Object snapshot(Tensor source) {
        if (HostStaging.owns(source)) {
            return source.dtype() == DType.FLOAT32 ? source.memory().floats() : source.memory().shorts();
        }
        int start = source.storageOffset(), end = start + source.numel();
        return source.dtype() == DType.FLOAT32
                ? Arrays.copyOfRange(source.memory().floats(), start, end)
                : Arrays.copyOfRange(source.memory().shorts(), start, end);
    }

    private static long pendingBytes(GraphCore graph) {
        long bytes = 0;
        for (Object data : graph.pendingUploadData) {
            if (data instanceof float[] values) bytes += (long) values.length * Float.BYTES;
            else if (data instanceof short[] values) bytes += (long) values.length * Short.BYTES;
        }
        return bytes;
    }
}
