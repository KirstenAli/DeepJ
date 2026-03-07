package io.github.kirstenali.deepj.data;

/**
 * One language-modeling batch: x are input token ids, y are targets (next-token ids).
 * Shapes:
 *  - x: [batchSize][seqLen]
 *  - y: [batchSize][seqLen]
 */
public record Batch(int[][] x, int[][] y) {}
