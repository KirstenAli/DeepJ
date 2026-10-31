package io.github.kirstenali.deepj.inference;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.transformer.embeddings.RotaryEmbedding;

import java.util.ArrayList;
import java.util.List;

/** Cache slots follow logical attention visits, so each repeated Loop pass has its own state. */
public final class InferenceContext implements AutoCloseable {
    private final int capacity;
    private final boolean rolling;
    private final List<Entry> entries = new ArrayList<>();
    private int position, cursor;

    public InferenceContext(int capacity) {
        this(capacity, false);
    }

    public InferenceContext(int capacity, boolean rolling) {
        if (capacity < 1) throw new IllegalArgumentException("Cache capacity must be positive");
        this.capacity = capacity;
        this.rolling = rolling;
    }

    public int position() { return position; }

    public void begin(int offset) {
        if (offset < 0 || (!rolling && offset >= capacity)) {
            throw new IllegalArgumentException("Invalid cache position");
        }
        position = offset;
        cursor = 0;
    }

    public Tensor rotate(RotaryEmbedding rotary, Tensor input, int length, int heads) {
        return rolling ? rotary.applyInference(input, length, heads, position)
                : rotary.apply(input, length, heads, position);
    }

    public Tensor attend(Object owner, Tensor query, Tensor key, Tensor value,
                          int queryHeads, int keyValueHeads, float scale) {
        if (GradMode.isEnabled()) throw new IllegalStateException("Inference caches require noGrad");
        CacheInputs input = CacheInputs.of(query, key, value, queryHeads, keyValueHeads);
        if (position > 0 && input.length() != 1) {
            throw new IllegalArgumentException("Incremental attention accepts one token at a time");
        }
        return slot(owner, input).attend(input, position, scale).reshape(query.shape());
    }

    private KeyValueCache slot(Object owner, CacheInputs input) {
        if (cursor == entries.size()) {
            if (position != 0) throw new IllegalStateException("Attention layout changed during generation");
            entries.add(new Entry(owner, new KeyValueCache(capacity, input)));
        }
        Entry entry = entries.get(cursor++);
        if (entry.owner() != owner) throw new IllegalStateException("Attention visit order changed");
        return entry.cache();
    }

    public void finish() {
        if (cursor != entries.size()) throw new IllegalStateException("Incomplete attention cache traversal");
    }

    @Override
    public void close() {
        entries.forEach(entry -> entry.cache().close());
        entries.clear();
        position = 0;
        cursor = 0;
    }

    private record Entry(Object owner, KeyValueCache cache) {}
}
