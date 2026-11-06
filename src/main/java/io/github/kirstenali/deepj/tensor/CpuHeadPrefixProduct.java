package io.github.kirstenali.deepj.tensor;

/** CPU cache prefixes retain their capacity stride instead of gathering and copying every cached token. */
public final class CpuHeadPrefixProduct {
    private CpuHeadPrefixProduct() {}

    public static Tensor apply(TensorBackend backend, Tensor left, Tensor cache,
                                int heads, int length, boolean transpose) {
        HeadCacheOps.requirePrefix(cache, heads, length);
        if (left.matrixRows() % heads != 0) throw new IllegalArgumentException("Invalid head query rows");
        int rows = left.matrixRows() / heads, capacity = cache.matrixRows() / heads;
        try (var scope = TensorScope.open()) {
            Tensor a = backend.reshape(left, heads, rows, left.matrixCols());
            Tensor full = backend.reshape(cache, heads, capacity, cache.matrixCols());
            Tensor prefix = backend.slice(full, 1, 0, length, 1);
            Tensor b = transpose ? backend.permute(prefix, 0, 2, 1) : prefix;
            Tensor result = backend.matmul(a, b);
            return scope.escape(backend.reshape(result, heads * rows, result.size(-1)));
        }
    }
}
