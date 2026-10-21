package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;

/** Runtime-only state; never serialized with a tensor or checkpoint. */
public final class GradientState {
    private boolean required;
    private long version;
    private Tensor gradient;
    private BackwardNode node;
    private final List<WeakReference<BackwardNode>> readers = new ArrayList<>();

    public boolean required() { return required; }
    public long version() { return version; }
    public Tensor gradient() { return gradient; }
    public BackwardNode node() { return node; }

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
        if (required && GradMode.isEnabled()) {
            throw new IllegalStateException("Use noGrad for parameter updates; tracked operations are out-of-place");
        }
        version++;
    }

    void accumulate(Tensor contribution) {
        if (gradient == null) gradient = Tensor.backend().addScalar(contribution, 0.0f);
        else {
            gradient.gradientState().beforeMutation();
            Tensor.backend().addInPlace(gradient, contribution);
        }
    }
}
