package io.github.kirstenali.deepj.data;

public record Batch(int[][] x, int[][] y, boolean[][] lossMask) {

    public Batch(int[][] x, int[][] y) {
        this(x, y, null);
    }

    public boolean[] mask(int row) {
        return lossMask == null ? null : lossMask[row];
    }
}
