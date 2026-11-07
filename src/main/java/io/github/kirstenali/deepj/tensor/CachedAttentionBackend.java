package io.github.kirstenali.deepj.tensor;

/**
 * Optional inference-only attention over a head-major key/value cache prefix.
 * Query rows group by cache head; key/value rows contain each head's full capacity.
 * The length selects only initialized prefix positions. Output has the query shape.
 */
public interface CachedAttentionBackend {
    boolean supportsCachedAttention(Tensor query, Tensor keys, Tensor values, int heads, int length, float scale);
    Tensor cachedAttention(Tensor query, Tensor keys, Tensor values, int heads, int length, float scale);
}
