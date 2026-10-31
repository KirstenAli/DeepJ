package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.tensor.Tensor;

/** Each response keeps its own token denominator; training averages these sequence objectives afterward. */
final class BatchedCausalLoss {
    private BatchedCausalLoss() {}

    static void requireBatch(Batch batch, int count) {
        if (batch == null || batch.x() == null || batch.y() == null
                || batch.x().length != count || batch.y().length != count) {
            throw new IllegalArgumentException("Dataset batch size must match the requested dense batch");
        }
        if (batch.lossMask() != null && batch.lossMask().length != count) {
            throw new IllegalArgumentException("Loss masks must match batch size");
        }
    }

    static Tensor sum(Tensor logits, Batch batch, int count) {
        if (logits.rank() != 3 || logits.size(0) != count) {
            throw new IllegalArgumentException("Batched logits must have shape[batch,time,vocabulary]");
        }
        Tensor total = null;
        for (int row = 0; row < count; row++) {
            Tensor loss = CrossEntropyLoss.forward(logits.select(0, row), batch.y()[row], batch.mask(row));
            total = total == null ? loss : total.add(loss);
        }
        return total;
    }
}
