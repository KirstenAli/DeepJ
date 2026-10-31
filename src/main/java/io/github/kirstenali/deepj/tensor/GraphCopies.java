package io.github.kirstenali.deepj.tensor;

/** Device-to-device raw copies; no host materialization of an existing GPU result. */
final class GraphCopies {
    private GraphCopies() {}

    static Tensor copy(GraphExecution graph, Tensor source) {
        TensorStorage.requireSource(source);
        if (source.numel() == 0) return Tensor.deviceOnly(source.dtype(), source.shape());
        if (!source.isContiguous()) return graph.transferLayout(source, source.dtype());
        if (source.getGpuTag() == null) return GraphHostCopies.copy(graph, source);
        GpuBuffer input = graph.ensureStorageBuffer(source);
        int rows = source.rank() == 2 ? source.matrixRows() : 1;
        GpuBuffer output = graph.newOutputBuffer(rows, source.numel() / rows, source.dtype());
        graph.recordStorageCopy(input, output, source.storageOffset(), 0, source.numel());
        return graph.createOutputLike(output, source);
    }

    static Tensor snapshotForWrite(GraphExecution graph, Tensor target, Tensor source) {
        if (target.memory() != source.memory()) return source;
        // The target is needed on device anyway; upload once, then snapshot it on the GPU.
        graph.ensureStorageBuffer(target);
        return copy(graph, source);
    }

    static void requireRange(GraphCore graph, GpuBuffer input, GpuBuffer output, int source, int target, int count) {
        input.requireLive();
        output.requireLive();
        if (input.owner != graph || output.owner != graph) {
            throw new IllegalArgumentException("Copy buffers must belong to the recording graph");
        }
        if (input.dtype != output.dtype || input.id == output.id) {
            throw new IllegalArgumentException("Raw GPU copies require matching dtypes and distinct allocations");
        }
        requireRegion(input, source, count);
        requireRegion(output, target, count);
    }

    private static void requireRegion(GpuBuffer buffer, int start, int count) {
        if (start < 0 || count < 1 || (long) start + count > buffer.elementCount()) {
            throw new IllegalArgumentException("GPU copy range exceeds its physical allocation");
        }
    }
}
