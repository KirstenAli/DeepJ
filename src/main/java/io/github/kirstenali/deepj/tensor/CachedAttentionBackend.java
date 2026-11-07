package io.github.kirstenali.deepj.tensor;

/**
 * Optional inference-only attention over a head-major key/value cache prefix.
 * Query rows group by cache head; key/value rows contain each head's full capacity.
 * The length selects only initialized prefix positions. Output has the query shape.
 */
public interface CachedAttentionBackend {
    /** Score scale, optional positive tanh softcap, and the first included key position. */
    record Options(float scale, float softcap, int firstPosition) {
        public Options {
            if (!Float.isFinite(scale) || !Float.isFinite(softcap) || softcap < 0 || firstPosition < 0) {
                throw new IllegalArgumentException("Expected finite attention scale, nonnegative softcap and position");
            }
        }
    }

    boolean supportsCachedAttention(Tensor query, Tensor keys, Tensor values, int heads, int length, float scale);
    Tensor cachedAttention(Tensor query, Tensor keys, Tensor values, int heads, int length, float scale);

    default boolean supportsCachedAttention(Tensor q, Tensor k, Tensor v, int heads, int length, Options options) {
        return options.softcap() == 0 && options.firstPosition() == 0
                && supportsCachedAttention(q, k, v, heads, length, options.scale());
    }

    default Tensor cachedAttention(Tensor q, Tensor k, Tensor v, int heads, int length, Options options) {
        if (!supportsCachedAttention(q, k, v, heads, length, options)) {
            throw new UnsupportedOperationException("Cached attention policy is unsupported");
        }
        return cachedAttention(q, k, v, heads, length, options.scale());
    }
}
