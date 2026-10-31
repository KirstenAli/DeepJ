package io.github.kirstenali.deepj.tensor;

/** Class scores occupy the final axis; every leading position has one integer target. */
public final class CrossEntropyChecks {
    private CrossEntropyChecks() {}

    public static int classes(Tensor logits) {
        if (logits == null || logits.rank() == 0 || logits.size(-1) == 0) {
            throw new IllegalArgumentException("Logits must have a nonempty final class axis");
        }
        return logits.size(-1);
    }

    public static int positions(Tensor logits) {
        int count = classes(logits);
        return logits.numel() / count;
    }

    public static void requireTargets(Tensor logits, int[] targets) {
        int classes = classes(logits), positions = positions(logits);
        if (targets == null || positions == 0 || targets.length != positions) {
            throw new IllegalArgumentException(
                    "CrossEntropy targets length must match nonempty leading logits positions");
        }
        for (int target : targets) {
            if (target < 0 || target >= classes) throw new IllegalArgumentException("Target class id is out of range");
        }
    }

    public static int included(boolean[] mask, int positions) {
        if (mask == null || mask.length != positions) {
            throw new IllegalArgumentException("Loss mask must match logits positions");
        }
        int result = 0;
        for (boolean value : mask) {
            if (value) result++;
        }
        if (result == 0) throw new IllegalArgumentException("Loss mask must include at least one position");
        return result;
    }
}
