package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.data.BatchSource;
import io.github.kirstenali.deepj.data.StatefulBatchSource;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.nn.transformer.CausalLM;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.Objects;

/** Preserves DeepJ's mean-per-sequence objective, including each sequence's own response mask denominator. */
public final class CausalObjective implements TrainingObjective {
    private final CausalLM model;
    private final BatchSource source;

    public CausalObjective(CausalLM model, BatchSource source) {
        this.model = Objects.requireNonNull(model, "model");
        this.source = Objects.requireNonNull(source, "source");
    }

    @Override public CausalLM model() { return model; }
    @Override public float gradientClipNorm() { return model.gradClipNorm(); }

    @Override public WorkerState workerState() {
        if (source instanceof WorkerState state) return state;
        return source instanceof StatefulBatchSource stateful ? new RandomCursorState(stateful) : WorkerState.empty();
    }

    @Override public WeightedLoss evaluate(int batchSize) {
        Batch batch = source.nextBatch(batchSize);
        if (batch == null || batch.x().length == 0) return WeightedLoss.empty();
        validate(batch);
        Tensor logits = model.supportsBatchedForward() ? model.forwardBatch(batch.x()) : null;
        Tensor total = null;
        for (int row = 0; row < batch.x().length; row++) {
            Tensor prediction = logits == null ? model.forward(batch.x()[row]) : logits.select(0, row);
            Tensor loss = CrossEntropyLoss.forward(prediction, batch.y()[row], batch.mask(row));
            total = total == null ? loss : total.add(loss);
        }
        return new WeightedLoss(total, batch.x().length);
    }

    private static void validate(Batch batch) {
        if (batch.y() == null || batch.x().length != batch.y().length
                || batch.lossMask() != null && batch.lossMask().length != batch.x().length) {
            throw new IllegalArgumentException("Token inputs, targets, and masks must have matching batch counts");
        }
    }
}
