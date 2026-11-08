package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import java.util.HashMap;
import java.util.Map;

/** Keeps repeated inference blocks within the existing working-weight budget without eviction thrashing. */
final class VideoBlockCache implements AutoCloseable {
    private final Map<String, TensorWeights> blocks = new HashMap<>();
    private long remaining;

    VideoBlockCache(long budget) { remaining = budget; }

    Selection borrow(String prefix) {
        TensorWeights weights = blocks.get(prefix);
        return weights == null ? null : new Selection(weights, false);
    }

    Selection keep(String prefix, TensorWeights weights) {
        long bytes = weights.descriptors().values().stream().mapToLong(descriptor -> descriptor.bytes()).sum();
        if (bytes > remaining) return new Selection(weights, true);
        blocks.put(prefix, weights);
        remaining -= bytes;
        return new Selection(weights, false);
    }

    record Selection(TensorWeights weights, boolean temporary) implements AutoCloseable {
        @Override public void close() {
            if (temporary) weights.close();
        }
    }

    @Override public void close() {
        blocks.values().forEach(TensorWeights::close);
        blocks.clear();
    }
}
