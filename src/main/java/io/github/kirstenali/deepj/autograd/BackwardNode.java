package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;

import java.util.Arrays;
import java.util.List;

final class BackwardNode {
    private List<Tensor> inputs;
    private List<SavedTensor> saved;
    private BackwardRule rule;
    private final TensorBackend backend = Tensor.backend();
    private boolean released;

    BackwardNode(BackwardRule rule, Tensor[] inputs, Tensor[] saved) {
        this.rule = rule;
        this.inputs = List.of(inputs);
        this.saved = Arrays.stream(saved).map(tensor -> new SavedTensor(tensor, this)).toList();
    }

    List<Tensor> inputs() { return inputs; }
    boolean released() { return released; }

    void validate() {
        if (released) throw new IllegalStateException("Backward graph was already released; run forward again");
        if (Tensor.backend() != backend) {
            throw new IllegalStateException("Backend changed between forward and backward");
        }
        saved.forEach(SavedTensor::validate);
    }

    Tensor[] apply(Tensor gradient) {
        Tensor[] results = rule.apply(gradient);
        if (results.length != inputs.size()) throw new IllegalStateException("Invalid backward rule arity");
        return results;
    }

    void release() {
        released = true;
        inputs = List.of();
        saved = List.of();
        rule = null;
    }
}
