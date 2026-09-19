package io.github.kirstenali.deepj.tokenizers.bpe;

import java.util.Arrays;

final class IntArrayBuilder {

    private int[] values;
    private int size;

    IntArrayBuilder(int expectedSize) {
        values = new int[Math.max(8, expectedSize)];
    }

    void add(int value) {
        ensureCapacity(size + 1);
        values[size++] = value;
    }

    void addAll(int[] source, int length) {
        ensureCapacity(size + length);
        System.arraycopy(source, 0, values, size, length);
        size += length;
    }

    int[] toArray() {
        return Arrays.copyOf(values, size);
    }

    private void ensureCapacity(int required) {
        if (required <= values.length) return;
        int capacity = Math.max(required, values.length * 2);
        values = Arrays.copyOf(values, capacity);
    }
}
