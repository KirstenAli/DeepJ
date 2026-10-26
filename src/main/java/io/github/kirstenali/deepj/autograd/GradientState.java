package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.DType;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

/** Runtime-only state; never serialized with a tensor or checkpoint. */
public final class GradientState {
    private boolean required;
    private long version;
    private MutationVersion sharedVersion;
    private Tensor gradient;
    private BackwardNode node;
    private final List<WeakReference<BackwardNode>> readers = new ArrayList<>();

    public boolean required() { return required; }
    public long version() { return sharedVersion == null ? version : sharedVersion.value(); }
    public Tensor gradient() { return gradient; }
    public BackwardNode node() { return node; }

    public GradientState() {}

    private GradientState(MutationVersion sharedVersion) {
        this.sharedVersion = sharedVersion;
        sharedVersion.track(this);
    }

    /** Independent autograd metadata sharing only the physical storage's mutation version. */
    public GradientState newAlias() {
        if (sharedVersion == null) {
            sharedVersion = new MutationVersion(version);
            sharedVersion.track(this);
        }
        return new GradientState(sharedVersion);
    }

    public boolean storageSavedForBackward() {
        return sharedVersion == null ? savedForBackward() : sharedVersion.any(GradientState::savedForBackward);
    }

    public void require(boolean value) {
        if (node != null) throw new IllegalStateException("Only leaf tensors can change requiresGrad");
        if (value != required && savedForBackward()) {
            throw new IllegalStateException("Cannot change requiresGrad while a backward graph uses this tensor");
        }
        required = value;
        if (!value) gradient = null;
    }

    public void bindGradient(Tensor value) { gradient = value; }

    void attach(BackwardNode value) {
        node = value;
        required = true;
    }

    void saveFor(BackwardNode reader) {
        pruneReaders();
        readers.add(new WeakReference<>(reader));
    }

    public boolean savedForBackward() {
        pruneReaders();
        return (node != null && !node.released()) || !readers.isEmpty();
    }

    private void pruneReaders() { readers.removeIf(GradientState::finished); }

    private static boolean finished(WeakReference<BackwardNode> reference) {
        BackwardNode node = reference.get();
        return node == null || node.released();
    }

    public void beforeMutation() {
        if (sharedVersion != null) {
            sharedVersion.beforeMutation();
            return;
        }
        if (required && GradMode.isEnabled()) {
            throw new IllegalStateException("Use noGrad for parameter updates; tracked operations are out-of-place");
        }
        version++;
    }

    void accumulate(Tensor contribution) {
        if (gradient == null) gradient = Tensor.backend().addScalar(contribution, 0.0f);
        else if (gradient.gradientState().savedForBackward()) gradient = Tensor.backend().add(gradient, contribution);
        else {
            gradient.gradientState().beforeMutation();
            Tensor.backend().addInPlace(gradient, contribution);
        }
    }

    void accumulateRows(int rows, int columns, int[] indices, Tensor contribution) {
        if (gradient == null) gradient = Tensor.zeros(rows, columns);
        if (gradient.matrixRows() != rows || gradient.matrixCols() != columns) {
            throw new IllegalArgumentException("Leaf gradient must match its tensor shape");
        }
        if (gradient.gradientState().savedForBackward()) gradient = Tensor.backend().addScalar(gradient, 0.0f);
        gradient.gradientState().beforeMutation();
        Tensor.backend().scatterAddRows(gradient, indices, contribution);
    }

    void accumulateIndexed(int[] shape, int axis, int[] indices, Tensor contribution) {
        if (gradient == null) gradient = Tensor.backend().zeros(DType.FLOAT32, shape);
        if (!java.util.Arrays.equals(shape, gradient.shape())) {
            throw new IllegalArgumentException("Indexed leaf gradient must match its tensor shape");
        }
        if (gradient.gradientState().savedForBackward()) gradient = Tensor.backend().addScalar(gradient, 0);
        gradient.gradientState().beforeMutation();
        Tensor.backend().indexAdd(gradient, axis, indices, contribution);
    }
}
