package io.github.kirstenali.deepj.tensor;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;

/** Per-flush immutable logical metadata; values and layout words use separate native buffers. */
final class GraphLayouts {
    private final Map<TensorLayout, Integer> offsets = new IdentityHashMap<>();
    private long[] words = new long[64];
    private int size;

    int add(Tensor tensor) {
        return offsets.computeIfAbsent(tensor.layout(), this::append);
    }

    int addIndices(int[] indices) {
        return addWords(IndexGroups.describe(indices));
    }

    int addWords(long[] description) {
        int start = size, needed = Math.addExact(size, description.length);
        if (needed > words.length) words = Arrays.copyOf(words, Math.max(needed, words.length * 2));
        System.arraycopy(description, 0, words, size, description.length);
        size = needed;
        return start;
    }

    private int append(TensorLayout layout) {
        int rank = layout.shape().rank(), start = size;
        int needed = Math.addExact(size, Math.addExact(4, Math.multiplyExact(rank, 2)));
        if (needed > words.length) words = Arrays.copyOf(words, Math.max(needed, words.length * 2));
        words[size++] = rank;
        words[size++] = layout.shape().elements();
        words[size++] = layout.storageElements();
        words[size++] = layout.offset();
        for (int dimension : layout.shape().dimensions()) { words[size++] = dimension; }
        for (long stride : layout.strides()) { words[size++] = stride; }
        return start;
    }

    boolean isEmpty() { return size == 0; }
    long[] snapshot() { return Arrays.copyOf(words, size); }

    void clear() {
        offsets.clear();
        size = 0;
        if (words.length > 4096) words = new long[64];
    }
}
