package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.DType;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;

/** Gemma layer weights use the same budgeted checkpoint cache as video blocks. Inference only. */
final class VideoTextWeights implements AutoCloseable {
    private final VideoWeightCache source;
    private TensorWeights weights;
    private String group = "";

    VideoTextWeights(Path directory) throws IOException {
        this(directory, DType.FLOAT32);
    }

    VideoTextWeights(Path directory, DType dtype) throws IOException {
        source = new VideoWeightCache(directory, VideoWeightCache.defaultBudget(), dtype);
    }

    Tensor tensor(String name) {
        String next = name.startsWith("layers.") ? name.substring(0, name.indexOf('.', 7)) + "." : name;
        if (!next.equals(group)) advance(next);
        return weights.tensor(name);
    }

    private void advance(String next) {
        // Gemma completes pending layer work before retiring its unretained weights at the block boundary.
        if (weights != null) weights.parameters().forEach(p -> p.value.releaseDeviceBufferRetention());
        try { weights = source.select(name -> name.startsWith(next)); }
        catch (IOException failure) { throw new UncheckedIOException(failure); }
        group = next;
    }

    @Override public void close() {
        if (weights != null) weights.close();
        source.close();
    }
}
