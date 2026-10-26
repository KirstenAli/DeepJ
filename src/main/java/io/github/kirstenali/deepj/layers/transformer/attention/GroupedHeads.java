package io.github.kirstenali.deepj.layers.transformer.attention;

import io.github.kirstenali.deepj.tensor.Tensor;

/** Expands shared KV heads and sums their gradients using the active tensor backend. */
final class GroupedHeads {

    private final int queryHeads;
    private final int keyValueHeads;
    private int[] mapping = new int[0];

    GroupedHeads(int queryHeads, int keyValueHeads) {
        if (keyValueHeads <= 0 || queryHeads % keyValueHeads != 0) {
            throw new IllegalArgumentException("Query heads must be divisible by KV heads");
        }
        this.queryHeads = queryHeads;
        this.keyValueHeads = keyValueHeads;
    }

    Tensor expand(Tensor heads, int sequenceLength) {
        prepareMapping(sequenceLength);
        return Tensor.sliceRows(heads, mapping, heads.matrixCols());
    }

    private void prepareMapping(int sequenceLength) {
        int rows = Math.multiplyExact(queryHeads, sequenceLength);
        if (mapping.length == rows) return;
        mapping = new int[rows];
        int groupSize = queryHeads / keyValueHeads;
        for (int row = 0; row < rows; row++) {
            int head = row / sequenceLength;
            mapping[row] = head / groupSize * sequenceLength + row % sequenceLength;
        }
    }
}
