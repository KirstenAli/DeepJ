package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.loss.LossFunction;
import java.util.Objects;

/** The supplied loss must be a local mean over samples (and any uniformly sized feature axes). */
public final class TensorObjective implements TrainingObjective {
    private final Layer model;
    private final LossFunction loss;
    private final TensorBatchSource source;
    private final float clipNorm;

    public TensorObjective(Layer model, LossFunction loss, TensorBatchSource source) {
        this(model, loss, source, Float.POSITIVE_INFINITY);
    }

    public TensorObjective(Layer model, LossFunction loss, TensorBatchSource source, float clipNorm) {
        this.model = Objects.requireNonNull(model, "model");
        this.loss = Objects.requireNonNull(loss, "loss");
        this.source = Objects.requireNonNull(source, "source");
        if (!(clipNorm > 0)) throw new IllegalArgumentException("Gradient norm must be positive");
        this.clipNorm = clipNorm;
    }

    @Override public Layer model() { return model; }
    @Override public float gradientClipNorm() { return clipNorm; }
    @Override public WorkerState workerState() {
        return source instanceof WorkerState state ? state : WorkerState.empty();
    }

    @Override public WeightedLoss evaluate(int batchSize) {
        TensorBatch batch = source.nextBatch(batchSize);
        if (batch == null) return WeightedLoss.empty();
        return WeightedLoss.mean(loss.forward(model.forward(batch.inputs()), batch.targets()), batch.samples());
    }
}
