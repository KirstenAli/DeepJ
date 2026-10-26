package io.github.kirstenali.deepj.tensor;

/** Unfused backend composition preserves the original matrix loss/gradient evaluation order. */
public final class CrossEntropyDefaults {
    private CrossEntropyDefaults() {}

    public static CrossEntropyResult apply(TensorBackend backend, Tensor logits, int[] targets, boolean[] mask) {
        Tensor loss = new Tensor(1, 1);
        loss.rawData()[0] = mask == null ? backend.crossEntropyLoss(logits, targets)
                : backend.crossEntropyLoss(logits, targets, mask);
        Tensor gradient = mask == null ? backend.crossEntropyGradient(logits, targets)
                : backend.crossEntropyGradient(logits, targets, mask);
        return new CrossEntropyResult(loss, gradient);
    }
}
