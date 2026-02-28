package org.DeepJ.ann.gpt;

/**
 * One language-modeling batch: x are input token ids, y are targets (next-token ids).
 * Shapes:
 *  - x: [batchSize][seqLen]
 *  - y: [batchSize][seqLen]
 */
public record Batch(int[][] x, int[][] y) {}
