package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.tensor.CachedAttentionBackend;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.HeadLayout;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Uses DeepJ's head-cache transfers and matrix products, with Gemma's score softcap and local mask. */
final class Gemma2Cache implements AutoCloseable {
    private final Gemma2Config config;
    private final Tensor[] keys, values;
    private final int capacity;
    private int position, length;

    Gemma2Cache(Gemma2Config config, int capacity) {
        this.config = config;
        this.capacity = capacity;
        keys = new Tensor[config.nLayers()];
        values = new Tensor[keys.length];
    }

    int position() { return position; }
    void finish(int count) { position += count; }

    void append(int layer, Tensor key, Tensor value) {
        length = Math.addExact(position, key.size(2));
        if (length > capacity) throw new IllegalArgumentException("Gemma cache capacity exceeded");
        if (keys[layer] == null) allocate(layer, key.dtype(), value.dtype());
        Tensor.backend().writeHeadCache(HeadLayout.pack(keys[layer]), HeadLayout.pack(key),
                config.kvHeads(), position);
        Tensor.backend().writeHeadCache(HeadLayout.pack(values[layer]), HeadLayout.pack(value),
                config.kvHeads(), position);
    }

    private void allocate(int layer, DType keyType, DType valueType) {
        keys[layer] = Tensor.backend().zeros(keyType, 1, config.kvHeads(), capacity, config.headDim())
                .retainDeviceBuffer();
        values[layer] = Tensor.backend().zeros(valueType, keys[layer].shape()).retainDeviceBuffer();
    }

    Tensor attend(int layer, Tensor query) {
        Tensor grouped = query.reshape(1, config.kvHeads(), config.nHeads() / config.kvHeads(), config.headDim());
        Tensor fused = fused(layer, HeadLayout.pack(grouped));
        if (fused != null) return fused.reshape(query.shape());
        Tensor scores = Tensor.backend().headPrefixMatmul(HeadLayout.pack(grouped),
                HeadLayout.pack(keys[layer]), config.kvHeads(), length, true);
        Tensor mask = layer % 2 == 0 && length > config.slidingWindow() ? localMask() : null;
        Tensor probabilities = Gemma2Attention.probabilities(scores, mask,
                (float) (1 / Math.sqrt(config.queryScalar())), config.attentionSoftcap()).to(values[layer].dtype());
        return Tensor.backend().headPrefixMatmul(probabilities, HeadLayout.pack(values[layer]),
                config.kvHeads(), length, false).reshape(query.shape());
    }

    private Tensor fused(int layer, Tensor query) {
        if (!(Tensor.backend() instanceof CachedAttentionBackend backend)) return null;
        int first = layer % 2 == 0 ? Math.max(0, length - config.slidingWindow()) : 0;
        var options = new CachedAttentionBackend.Options((float) (1 / Math.sqrt(config.queryScalar())),
                config.attentionSoftcap(), first);
        Tensor key = HeadLayout.pack(keys[layer]), value = HeadLayout.pack(values[layer]);
        return backend.supportsCachedAttention(query, key, value, config.kvHeads(), length, options)
                ? backend.cachedAttention(query, key, value, config.kvHeads(), length, options) : null;
    }

    private Tensor localMask() {
        float[] bias = new float[length];
        java.util.Arrays.fill(bias, 0, length - config.slidingWindow(), -1e30f);
        return Tensor.from(bias, 1, length);
    }

    @Override public void close() {
        for (int layer = 0; layer < keys.length; layer++) {
            if (keys[layer] != null) keys[layer].releaseDeviceBufferRetention();
            if (values[layer] != null) values[layer].releaseDeviceBufferRetention();
        }
    }
}
