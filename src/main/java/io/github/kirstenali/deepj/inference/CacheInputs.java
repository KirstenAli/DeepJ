package io.github.kirstenali.deepj.inference;

import io.github.kirstenali.deepj.tensor.HeadLayout;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Distinct batch/head/time axes; legacy packed matrices describe one sequence only. */
record CacheInputs(Tensor query, Tensor key, Tensor value, int queryHeads, int keyValueHeads) {
    static CacheInputs of(Tensor query, Tensor key, Tensor value, int queryHeads, int keyValueHeads) {
        requireHeads(queryHeads, keyValueHeads);
        Tensor q = heads(query, queryHeads), k = heads(key, keyValueHeads), v = heads(value, keyValueHeads);
        Tensor.requireSameShape(k, v, "Cached values");
        if (k.size(0) < 1 || k.size(2) < 1 || q.size(0) != k.size(0) || q.size(2) != k.size(2)
                || q.size(3) != k.size(3)) {
            throw new IllegalArgumentException("Cached attention needs matching nonempty batch,time and head width");
        }
        return new CacheInputs(q, k, v, queryHeads, keyValueHeads);
    }

    private static void requireHeads(int queryHeads, int keyValueHeads) {
        if (queryHeads < 1 || keyValueHeads < 1 || queryHeads % keyValueHeads != 0) {
            throw new IllegalArgumentException("Query heads must be a positive multiple of key/value heads");
        }
    }

    private static Tensor heads(Tensor input, int heads) {
        if (input.rank() == 2) {
            if (input.matrixRows() % heads != 0 || input.matrixCols() < 1) {
                throw new IllegalArgumentException("Invalid packed cache heads");
            }
            return input.reshape(1, heads, input.matrixRows() / heads, input.matrixCols());
        }
        HeadLayout.require(input, heads);
        return input;
    }

    int batches() { return key.size(0); }
    int length() { return key.size(2); }
    int width() { return key.size(3); }
}
