package io.github.kirstenali.deepj.tensor;

/** Inference-only storage operations for head-major key/value caches. */
public final class HeadCacheOps {
    private HeadCacheOps() {}

    public static void requireWrite(Tensor cache, Tensor update, int heads, int offset) {
        requirePrefix(cache, heads, 1);
        if (update.matrixRows() % heads != 0 || update.matrixCols() != cache.matrixCols() || offset < 0
                || (long) offset + update.matrixRows() / heads > cache.matrixRows() / heads) {
            throw new IllegalArgumentException("Head cache write exceeds its shape or capacity");
        }
        cache.layout().requireWritable();
        if (cache.requiresGrad() || cache.gradientState().storageSavedForBackward()) {
            throw new IllegalArgumentException("Head caches must be detached scratch tensors");
        }
    }

    public static void requirePrefix(Tensor cache, int heads, int length) {
        if (heads < 1 || cache.matrixRows() % heads != 0 || length < 1 || length > cache.matrixRows() / heads) {
            throw new IllegalArgumentException("Invalid head cache prefix");
        }
    }

    static void write(TensorBackend backend, Tensor cache, Tensor update, int heads, int offset) {
        if (!CpuNdElementwise.matrix(cache) || !CpuNdElementwise.matrix(update)) {
            writeLayouts(backend, cache, update, heads, offset);
            return;
        }
        requireWrite(cache, update, heads, offset);
        cache.gradientState().beforeMutation();
        cache.materialize();
        update.materialize();
        int count = update.matrixRows() / heads * update.matrixCols();
        for (int head = 0; head < heads; head++) {
            int destination = (head * (cache.matrixRows() / heads) + offset) * cache.matrixCols();
            System.arraycopy(update.rawData(), head * count, cache.rawData(), destination, count);
        }
        cache.setGpuTag(null);
    }

    public static void writeLayouts(TensorBackend backend, Tensor cache, Tensor update, int heads, int offset) {
        requireWrite(cache, update, heads, offset);
        cache.gradientState().beforeMutation();
        Tensor source = cache.memory() == update.memory() ? backend.copy(update) : update;
        int count = update.matrixRows() / heads, capacity = cache.matrixRows() / heads;
        for (int head = 0; head < heads; head++) {
            Tensor target = backend.slice(cache, 0, head * capacity + offset, head * capacity + offset + count, 1);
            backend.copyInto(target, backend.slice(source, 0, head * count, (head + 1) * count, 1));
        }
        if (cache.getGpuTag() instanceof GpuBuffer buffer && !buffer.cpuStale) buffer.needsUpload = true;
    }

    public static Tensor prefix(TensorBackend backend, Tensor cache, int heads, int length) {
        requirePrefix(cache, heads, length);
        int[] rows = new int[heads * length];
        for (int row = 0; row < rows.length; row++) {
            rows[row] = row / length * (cache.matrixRows() / heads) + row % length;
        }
        return backend.sliceRows(cache, rows);
    }
}
