package io.github.kirstenali.deepj.transformer.embeddings;

/** Rectangular integer token metadata is copied before gathering any numerical values. */
record TokenBatch(int batches, int length, int[] tokens) {
    static TokenBatch from(int[][] sequences) {
        if (sequences == null || sequences.length == 0 || sequences[0] == null || sequences[0].length == 0) {
            throw new IllegalArgumentException("Token batches require nonempty equal-length sequences");
        }
        int length = sequences[0].length;
        requireRows(sequences, length);
        int[] tokens = new int[Math.multiplyExact(sequences.length, length)];
        for (int row = 0; row < sequences.length; row++) {
            System.arraycopy(sequences[row], 0, tokens, row * length, length);
        }
        return new TokenBatch(sequences.length, length, tokens);
    }

    private static void requireRows(int[][] sequences, int length) {
        for (int[] sequence : sequences) {
            if (sequence == null || sequence.length != length) {
                throw new IllegalArgumentException("Dense token batches cannot contain ragged or null sequences");
            }
        }
    }
}
