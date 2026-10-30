package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

import java.util.List;

final class BackwardNode {
    private List<GradientEdge> inputs;
    private List<SavedTensor> saved;
    private BackwardRule rule;
    private final TensorBackend backend = Tensor.backend();
    private boolean released;
    private final boolean builtin;

    BackwardNode(BackwardRule rule, Tensor[] inputs, Tensor[] saved, boolean builtin) {
        this.rule = rule;
        this.builtin = builtin;
        this.inputs = edges(inputs);
        for (int index = 0; index < this.inputs.size(); index++) {
            this.inputs.get(index).state().saveFor(this);
        }
        this.saved = saved(saved);
    }

    private static List<GradientEdge> edges(Tensor[] inputs) {
        GradientEdge[] result = new GradientEdge[inputs.length];
        for (int index = 0; index < inputs.length; index++) { result[index] = new GradientEdge(inputs[index]); }
        return List.of(result);
    }

    private List<SavedTensor> saved(Tensor[] tensors) {
        SavedTensor[] result = new SavedTensor[tensors.length];
        for (int index = 0; index < tensors.length; index++) { result[index] = new SavedTensor(tensors[index], this); }
        return List.of(result);
    }

    List<GradientEdge> inputs() { return inputs; }
    boolean released() { return released; }
    boolean isBuiltin() { return builtin || rule instanceof GatherBackward || rule instanceof IndexSelectBackward; }

    void validate() {
        if (released) throw new IllegalStateException("Backward graph was already released; run forward again");
        if (Tensor.backend() != backend) {
            throw new IllegalStateException("Backend changed between forward and backward");
        }
        inputs.forEach(GradientEdge::validate);
        saved.forEach(SavedTensor::validate);
    }

    Tensor[] apply(Tensor gradient) {
        Tensor[] results = rule.apply(gradient);
        if (results.length != inputs.size()) throw new IllegalStateException("Invalid backward rule arity");
        return results;
    }

    boolean accumulateLeaf(Tensor gradient) {
        return rule instanceof LeafGradientRule leaf && leaf.accumulate(gradient);
    }

    void release() {
        released = true;
        inputs = List.of();
        saved = List.of();
        rule = null;
    }
}
