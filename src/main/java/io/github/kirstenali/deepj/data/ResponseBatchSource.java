package io.github.kirstenali.deepj.data;

import java.util.Arrays;

abstract class ResponseBatchSource implements StatefulBatchSource {

    private final StatefulRandom random;

    ResponseBatchSource(long seed) {
        this.random = new StatefulRandom(seed);
    }

    @Override
    public final synchronized Batch nextBatch(int batchSize) {
        if (batchSize < 1) throw new IllegalArgumentException("batchSize must be positive");
        int[][] inputs = new int[batchSize][];
        int[][] targets = new int[batchSize][];
        boolean[][] masks = new boolean[batchSize][];
        for (int row = 0; row < batchSize; row++) {
            sampleRow(inputs, targets, masks, row);
        }
        return new Batch(inputs, targets, masks);
    }

    abstract void sampleRow(int[][] inputs, int[][] targets, boolean[][] masks, int row);

    final StatefulRandom random() {
        return random;
    }

    @Override
    public final synchronized long randomState() {
        return random.state();
    }

    @Override
    public final synchronized void restoreRandomState(long state) {
        random.restore(state);
    }

    static void fill(int[] tokens, int responseStart, int[][] inputs,
                     int[][] targets, boolean[][] masks, int row) {
        int length = tokens.length - 1;
        inputs[row] = Arrays.copyOf(tokens, length);
        targets[row] = Arrays.copyOfRange(tokens, 1, length + 1);
        masks[row] = mask(length, responseStart);
    }

    private static boolean[] mask(int length, int responseStart) {
        boolean[] mask = new boolean[length];
        Arrays.fill(mask, responseStart - 1, length, true);
        return mask;
    }
}
