package io.github.kirstenali.deepj.inference;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Fixed-capacity head-major scratch storage; never part of a checkpoint or backward graph. */
final class KeyValueCache implements AutoCloseable {
    private final Tensor keys, values;
    private final int heads;
    private int length;

    KeyValueCache(int capacity, int heads, int width) {
        this.heads = heads;
        keys = new Tensor(Math.multiplyExact(capacity, heads), width).retainDeviceBuffer();
        values = new Tensor(keys.rows, width).retainDeviceBuffer();
    }

    Tensor attend(Tensor query, Tensor key, Tensor value, int queryHeads, int offset, float scale) {
        if (offset != length) throw new IllegalStateException("Head cache position mismatch");
        Tensor.backend().writeHeadCache(keys, key, heads, offset);
        Tensor.backend().writeHeadCache(values, value, heads, offset);
        length += key.rows / heads;
        if (offset == 0) return AttentionOps.causal(query, key, value, queryHeads, heads, scale);
        return decode(query, scale);
    }

    private Tensor decode(Tensor query, float scale) {
        Tensor scores = Tensor.backend().headPrefixMatmul(query, keys, heads, length, true);
        Tensor probabilities = Tensor.backend().softmaxRows(scores.multiplyScalar(scale));
        return Tensor.backend().headPrefixMatmul(probabilities, values, heads, length, false);
    }

    @Override
    public void close() {
        keys.releaseDeviceBufferRetention();
        values.releaseDeviceBufferRetention();
    }
}
