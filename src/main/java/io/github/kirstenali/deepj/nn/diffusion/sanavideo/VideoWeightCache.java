package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Predicate;

/** Budgeted checkpoint storage with FP32 working values. Inference only. */
final class VideoWeightCache implements AutoCloseable {
    private static final long GIB = 1024L * 1024 * 1024;
    private final Path directory;
    private final Map<String, TensorDescriptor> descriptors;
    private final Map<DType, TensorWeights> sources = new EnumMap<>(DType.class);
    private final boolean cached;
    private boolean closed;

    VideoWeightCache(Path directory, long budget) throws IOException {
        if (budget < 0) throw new IllegalArgumentException("Video weight cache budget must be nonnegative");
        this.directory = directory;
        try (var metadata = TensorWeights.open(directory, DType.FLOAT32)) {
            descriptors = metadata.descriptors();
        }
        cached = descriptors.values().stream().mapToLong(TensorDescriptor::bytes).sum() <= budget;
    }

    static long defaultBudget() {
        long available = Tensor.backend() instanceof MetalBackend
                ? Long.parseLong(System.getProperty("deepj.metal.maxBufferBytes", Long.toString(6 * GIB)))
                : Runtime.getRuntime().maxMemory();
        long automatic = Tensor.backend() instanceof MetalBackend
                ? Math.min(4 * GIB, available / 2) : available / 4;
        return Long.parseLong(System.getProperty("deepj.video.weightCacheBytes", Long.toString(automatic)));
    }

    boolean cached() { return cached; }

    TensorWeights select(Predicate<String> selected) throws IOException {
        if (closed) throw new IllegalStateException("Video weights are closed");
        TensorWeights stream = cached ? null : TensorWeights.open(directory, DType.FLOAT32);
        try {
            var tensors = new LinkedHashMap<String, Tensor>();
            descriptors.keySet().stream().filter(selected).sorted()
                    .forEach(name -> tensors.put(name, read(name, stream)));
            return TensorWeights.from(tensors);
        } finally {
            // The caller may still have pending activations. Its block boundary performs temporary cleanup.
            if (stream != null) stream.parameters().forEach(p -> p.value.releaseDeviceBufferRetention());
        }
    }

    private Tensor read(String name, TensorWeights stream) {
        Tensor value = stream == null ? source(descriptors.get(name).dtype()).tensor(name) : stream.tensor(name);
        Tensor alias = Tensor.backend().expand(value, value.shape());
        return alias.to(DType.FLOAT32);
    }

    private TensorWeights source(DType dtype) {
        return sources.computeIfAbsent(dtype, type -> {
            try { return TensorWeights.open(directory, type); }
            catch (IOException failure) { throw new UncheckedIOException(failure); }
        });
    }

    @Override public void close() {
        sources.values().forEach(TensorWeights::close);
        sources.clear();
        closed = true;
    }
}
