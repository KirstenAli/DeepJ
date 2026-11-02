package io.github.kirstenali.deepj.distributed;

import io.github.kirstenali.deepj.optimisers.ParameterOptimizer;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.training.Trainer;
import java.util.Objects;

/** A separate training entry point; no existing DeepJ trainer, optimizer, model, or backend is patched. */
public final class DistributedTrainer implements AutoCloseable {
    final ProcessGroup group;
    final TrainingObjective objective;
    final ParameterOptimizer optimizer;
    final DistributedOptions options;
    final ParameterLayout layout;
    private final GradientReducer reducer;
    private final float clipNorm;
    private boolean initialized;
    private byte[] signature;
    private DistributedStep last = new DistributedStep(0, 0, Float.NaN);

    public DistributedTrainer(ProcessGroup group, TrainingObjective objective, ParameterOptimizer optimizer) {
        this(group, objective, optimizer, DistributedOptions.defaults());
    }

    public DistributedTrainer(ProcessGroup group, TrainingObjective objective,
                              ParameterOptimizer optimizer, DistributedOptions options) {
        this.group = Objects.requireNonNull(group, "group");
        this.objective = Objects.requireNonNull(objective, "objective");
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        this.options = Objects.requireNonNull(options, "options");
        if (!(objective.gradientClipNorm() > 0)) throw new IllegalArgumentException("Invalid gradient norm");
        clipNorm = objective.gradientClipNorm();
        layout = new ParameterLayout(objective);
        reducer = new GradientReducer(group, options.bucketElements());
    }

    public synchronized void initialize() {
        if (initialized) return;
        signature = layout.manifest(objective, optimizer, options);
        RankAgreement.same(group, "model.layout", signature);
        TensorTransfer.broadcast(group, "model.weights", layout.parameters.stream().map(p -> p.value).toList(),
                options.bucketElements());
        broadcastBuffers();
        initialized = true;
    }

    public Trainer trainer() { return new Trainer(this::trainStep); }
    public DistributedStep lastStep() { return last; }
    public float trainStep(int localBatchSize) { return step(localBatchSize).loss(); }

    public synchronized DistributedStep step(int localBatchSize) {
        try (var scope = TensorScope.open()) {
            initialize();
            WeightedLoss local = evaluate(localBatchSize);
            StepTotals total = StepTotals.collect(group, last.step(), local, optimizer);
            backward(local);
            reducer.average(layout.parameters, total.weight());
            reducer.clip(layout.parameters, clipNorm);
            update();
            if (options.bufferPolicy() == BufferPolicy.BROADCAST_FROM_ROOT) broadcastBuffers();
            last = new DistributedStep(last.step() + 1, total.weight(), total.meanLoss());
            return last;
        } catch (RuntimeException failure) {
            group.abort(failure.getMessage());
            throw failure;
        }
    }

    private WeightedLoss evaluate(int batchSize) {
        WeightedLoss loss = null;
        Exception failure = null;
        try {
            if (batchSize < 1) throw new IllegalArgumentException("Local batch size must be positive");
            layout.verify(objective);
            verifySettings();
            objective.model().zeroGrad();
            loss = objective.evaluate(batchSize);
            float value = loss.sum().item();
            if (!Float.isFinite(value) || loss.weight() == 0 && value != 0) {
                throw new IllegalStateException("Non-finite or invalid empty loss");
            }
        } catch (RuntimeException error) { failure = error; }
        RankAgreement.success(group, "step.forward", failure);
        return loss;
    }

    private void verifySettings() {
        if (objective.gradientClipNorm() != clipNorm) throw new IllegalStateException("Gradient norm setting changed");
        if (optimizer instanceof io.github.kirstenali.deepj.optimisers.AdamW adam && adam.stepCount() != last.step()) {
            throw new IllegalStateException("Optimizer advanced outside the distributed trainer");
        }
    }

    private void backward(WeightedLoss loss) {
        Exception failure = null;
        try {
            if (loss.weight() > 0 && loss.sum().requiresGrad()) loss.sum().backward();
        } catch (RuntimeException error) { failure = error; }
        RankAgreement.success(group, "step.backward", failure);
    }

    private void update() {
        Exception failure = null;
        try { optimizer.step(layout.parameters); }
        catch (RuntimeException error) { failure = error; }
        RankAgreement.success(group, "step.optimizer", failure);
    }

    private void broadcastBuffers() {
        TensorTransfer.broadcast(group, "model.buffers", layout.buffers, options.bucketElements());
    }

    void restoreProgress(DistributedStep step) { last = step; }
    byte[] signature() { return signature.clone(); }
    @Override public void close() { group.close(); }
}
