package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class BackwardPass {
    private final List<Tensor> order = new ArrayList<>();
    private final Map<Tensor, Tensor> gradients = new IdentityHashMap<>();

    BackwardPass(Tensor output, Tensor upstream) {
        visit(output);
        gradients.put(output, upstream);
    }

    private void visit(Tensor root) {
        Set<Tensor> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        var pending = new ArrayDeque<Visit>();
        pending.push(new Visit(root, false));
        while (!pending.isEmpty()) {
            Visit next = pending.pop();
            if (next.expanded()) order.add(next.tensor());
            else if (seen.add(next.tensor())) expand(next.tensor(), pending);
        }
    }

    private void expand(Tensor tensor, ArrayDeque<Visit> pending) {
        pending.push(new Visit(tensor, true));
        BackwardNode node = tensor.gradientState().node();
        if (node == null) return;
        for (Tensor input : node.inputs()) {
            if (input.requiresGrad()) pending.push(new Visit(input, false));
        }
    }

    void run() {
        try {
            validate();
            for (int index = order.size() - 1; index >= 0; index--) {
                propagate(order.set(index, null));
            }
        } finally {
            release();
        }
    }

    private void validate() {
        for (Tensor tensor : order) {
            BackwardNode node = tensor.gradientState().node();
            if (node != null) node.validate();
        }
    }

    private void propagate(Tensor tensor) {
        Tensor gradient = gradients.remove(tensor);
        if (gradient == null) return;
        BackwardNode node = tensor.gradientState().node();
        if (node == null) tensor.gradientState().accumulate(gradient);
        else propagateInputs(node, gradient);
    }

    private void propagateInputs(BackwardNode node, Tensor gradient) {
        try {
            Tensor[] contributions = node.apply(gradient);
            for (int index = 0; index < contributions.length; index++) {
                accumulate(node.inputs().get(index), contributions[index]);
            }
        } finally {
            node.release();
        }
    }

    private void accumulate(Tensor input, Tensor contribution) {
        if (!input.requiresGrad()) return;
        if (contribution == null) throw new IllegalStateException("Missing gradient for a tracked input");
        Tensor.requireSameShape(input, contribution, "operation gradient");
        gradients.merge(input, contribution, (left, right) -> Tensor.backend().add(left, right));
    }

    private void release() {
        for (Tensor tensor : order) {
            if (tensor == null) continue;
            BackwardNode node = tensor.gradientState().node();
            if (node != null) node.release();
        }
        gradients.clear();
    }

    private record Visit(Tensor tensor, boolean expanded) {}
}
