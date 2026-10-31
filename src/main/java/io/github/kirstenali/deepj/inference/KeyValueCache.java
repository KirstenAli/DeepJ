package io.github.kirstenali.deepj.inference;

import io.github.kirstenali.deepj.autograd.AttentionOps;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.HeadLayout;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Fixed-capacity head-major scratch storage; never part of a checkpoint or backward graph. */
final class KeyValueCache implements AutoCloseable {
    private final Tensor keys, values;
    private final int heads, capacity;
    private int length, processed;

    KeyValueCache(int capacity, int heads, int width) {
        this(capacity, 1, heads, width, DType.FLOAT32, DType.FLOAT32);
    }

    KeyValueCache(int capacity, CacheInputs input) {
        this(capacity, input.batches(), input.keyValueHeads(), input.width(),
                input.key().dtype(), input.value().dtype());
    }

    private KeyValueCache(int capacity, int batch, int heads, int width, DType keyType, DType valueType) {
        if (capacity < 1 || batch < 1 || heads < 1 || width < 1) {
            throw new IllegalArgumentException("Cache axes must be positive");
        }
        this.heads = Math.multiplyExact(batch, heads);
        this.capacity = capacity;
        keys = Tensor.backend().zeros(keyType, batch, heads, capacity, width).retainDeviceBuffer();
        values = Tensor.backend().zeros(valueType, batch, heads, capacity, width).retainDeviceBuffer();
    }

    Tensor attend(Tensor query, Tensor key, Tensor value, int queryHeads, int offset, float scale) {
        CacheInputs input = CacheInputs.of(query, key, value, queryHeads, keys.size(1));
        return attend(input, offset, scale).reshape(query.shape());
    }

    Tensor attend(CacheInputs input, int offset, float scale) {
        requireInput(input, offset, scale);
        append(input, offset);
        return offset == 0 ? AttentionOps.causal(input.query(), input.key(), input.value(),
                input.queryHeads(), input.keyValueHeads(), scale) : decode(input.query(), scale);
    }

    private void requireInput(CacheInputs input, int offset, float scale) {
        if (offset != processed) throw new IllegalStateException("Head cache position mismatch");
        requireStorage(input);
        if (input.length() > capacity || (offset > 0 && input.length() != 1) || !Float.isFinite(scale)) {
            throw new IllegalArgumentException("Invalid cached attention scale or incremental length");
        }
    }

    private void requireStorage(CacheInputs input) {
        if (input.batches() != keys.size(0) || input.keyValueHeads() != keys.size(1) || input.width() != keys.size(3)
                || input.key().dtype() != keys.dtype() || input.value().dtype() != values.dtype()) {
            throw new IllegalArgumentException("Cached attention layout or storage dtype changed");
        }
    }

    private void append(CacheInputs input, int offset) {
        int next = Math.addExact(offset, input.length());
        Tensor.backend().writeHeadCache(HeadLayout.pack(keys), HeadLayout.pack(input.key()), heads, offset % capacity);
        Tensor.backend().writeHeadCache(HeadLayout.pack(values), HeadLayout.pack(input.value()),
                heads, offset % capacity);
        processed = next;
        length = Math.min(capacity, processed);
    }

    private Tensor decode(Tensor query, float scale) {
        Tensor scores = Tensor.backend().headPrefixMatmul(HeadLayout.pack(query), HeadLayout.pack(keys),
                heads, length, true);
        Tensor probabilities = Tensor.backend().softmaxRows(scores.multiplyScalar(scale));
        return Tensor.backend().headPrefixMatmul(probabilities, HeadLayout.pack(values), heads, length, false)
                .reshape(query.shape());
    }

    @Override
    public void close() {
        keys.releaseDeviceBufferRetention();
        values.releaseDeviceBufferRetention();
    }
}
