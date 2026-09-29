package io.github.kirstenali.deepj.data;

final class CausalBatches {

    private CausalBatches() {}

    static Batch from(int[] tokens, int batchSize, int sequenceLength) {
        int[][] inputs = new int[batchSize][sequenceLength];
        int[][] targets = new int[batchSize][sequenceLength];
        for (int row = 0; row < batchSize; row++) {
            fillRow(tokens, inputs[row], targets[row], row * sequenceLength);
        }
        return new Batch(inputs, targets);
    }

    private static void fillRow(int[] tokens, int[] inputs, int[] targets, int offset) {
        System.arraycopy(tokens, offset, inputs, 0, inputs.length);
        System.arraycopy(tokens, offset + 1, targets, 0, targets.length);
    }
}
