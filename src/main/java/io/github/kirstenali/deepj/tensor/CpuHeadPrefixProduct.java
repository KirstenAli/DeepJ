package io.github.kirstenali.deepj.tensor;

/** CPU entry point for the shared capacity-strided head-cache product. */
public final class CpuHeadPrefixProduct {
    private CpuHeadPrefixProduct() {}

    public static Tensor apply(TensorBackend backend, Tensor left, Tensor cache,
                                int heads, int length, boolean transpose) {
        return HeadCacheOps.product(backend, left, cache, heads, length, transpose);
    }
}
