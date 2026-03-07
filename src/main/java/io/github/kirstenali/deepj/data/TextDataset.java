package io.github.kirstenali.deepj.data;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * Simple in-memory dataset that samples random contiguous chunks from token ids.
 * This is intentionally minimal but correct; for large corpora, replace with a
 * memory-mapped or streaming implementation.
 */
public final class TextDataset {

    private final int[] tokens;
    private final int seqLen;
    private final Random rnd;

    public TextDataset(int[] tokens, int seqLen, long seed) {
        if (seqLen < 2) throw new IllegalArgumentException("seqLen must be >= 2");
        if (tokens.length < seqLen + 1) throw new IllegalArgumentException("Not enough tokens for seqLen+1");
        this.tokens = tokens;
        this.seqLen = seqLen;
        this.rnd = new Random(seed);
    }

    public static TextDataset fromFile(Path path, Tokenizer tok, int seqLen, long seed) throws IOException {
        String text = Files.readString(path);
        return new TextDataset(tok.encode(text), seqLen, seed);
    }

    public Batch nextBatch(int batchSize) {
        int[][] x = new int[batchSize][seqLen];
        int[][] y = new int[batchSize][seqLen];

        int maxStart = tokens.length - (seqLen + 1);

        for (int b = 0; b < batchSize; b++) {
            int start = rnd.nextInt(maxStart);
            for (int t = 0; t < seqLen; t++) {
                x[b][t] = tokens[start + t];
                y[b][t] = tokens[start + t + 1];
            }
        }
        return new Batch(x, y);
    }

    public int seqLen() {
        return seqLen;
    }

    public int size() {
        return tokens.length;
    }
}
