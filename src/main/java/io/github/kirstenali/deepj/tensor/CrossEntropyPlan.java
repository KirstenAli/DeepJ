package io.github.kirstenali.deepj.tensor;

/** Immutable exact integer targets and row inclusion bits in the graph's metadata stream. */
final class CrossEntropyPlan {
    private CrossEntropyPlan() {}

    static long[] describe(Tensor logits, int[] targets, boolean[] mask) {
        int[] ids = targets == null ? null : targets.clone();
        boolean[] included = mask == null ? null : mask.clone();
        CrossEntropyChecks.requireTargets(logits, ids);
        int rows = ids.length, count = included == null ? rows : CrossEntropyChecks.included(included, rows);
        long[] words = new long[Math.addExact(3, Math.multiplyExact(rows, 2))];
        words[0] = rows;
        words[1] = count;
        words[2] = included == null ? 0 : 1;
        for (int row = 0; row < rows; row++) {
            words[3 + row] = ids[row];
            words[3 + rows + row] = included == null || included[row] ? 1 : 0;
        }
        return words;
    }
}
