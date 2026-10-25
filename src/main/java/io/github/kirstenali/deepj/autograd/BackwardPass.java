package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.BackwardBuffers;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class BackwardPass {
    private final List<GradientEdge> order = new ArrayList<>();
    private final Map<GradientState, Tensor> gradients = new IdentityHashMap<>();
    private BackwardBuffers buffers;
    private final boolean privateBuffers = Tensor.backend() instanceof MetalBackend;

    BackwardPass(Tensor output, Tensor upstream) {
        visit(new GradientEdge(output));
        gradients.put(output.gradientState(), upstream);
    }

    private void visit(GradientEdge root) {
        Set<GradientState> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        var pending = new ArrayDeque<Visit>();
        pending.push(new Visit(root, false));
        while (!pending.isEmpty()) {
            Visit next = pending.pop();
            if (next.expanded()) order.add(next.edge());
            else if (seen.add(next.edge().state())) expand(next.edge(), pending);
        }
    }

    private void expand(GradientEdge edge, ArrayDeque<Visit> pending) {
        pending.push(new Visit(edge, true));
        BackwardNode node = edge.state().node();
        if (node == null) return;
        for (GradientEdge input : node.inputs()) {
            if (input.state().required()) pending.push(new Visit(input, false));
        }
    }

    void run() {
        if (!privateBuffers) {
            execute();
            return;
        }
        try (var scratch = BackwardBuffers.begin()) {
            buffers = scratch;
            execute();
        }
    }

    private void execute() {
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
        for (GradientEdge edge : order) {
            BackwardNode node = edge.state().node();
            if (node != null) node.validate();
        }
    }

    private void propagate(GradientEdge edge) {
        Tensor gradient = gradients.remove(edge.state());
        if (gradient == null) return;
        BackwardNode node = edge.state().node();
        if (node == null) edge.state().accumulate(gradient);
        else propagateInputs(node, gradient);
    }

    private void propagateInputs(BackwardNode node, Tensor gradient) {
        try {
            if (node.accumulateLeaf(gradient)) return;
            Tensor[] contributions = apply(node, gradient);
            for (int index = 0; index < contributions.length; index++) {
                accumulate(node.inputs().get(index), contributions[index]);
            }
        } finally {
            node.release();
        }
    }

    private Tensor[] apply(BackwardNode node, Tensor gradient) {
        if (privateBuffers && node.isBuiltin()) return buffers.capture(() -> node.apply(gradient));
        if (privateBuffers) buffers.expose(gradient);
        return node.apply(gradient);
    }

    private void accumulate(GradientEdge input, Tensor contribution) {
        if (!input.state().required()) return;
        if (contribution == null) throw new IllegalStateException("Missing gradient for a tracked input");
        input.requireShape(contribution);
        if (input.state().node() == null) input.state().accumulate(contribution);
        else gradients.merge(input.state(), contribution, this::combine);
    }

    private Tensor combine(Tensor left, Tensor right) {
        return privateBuffers ? buffers.capture(() -> Tensor.backend().add(left, right))
                : Tensor.backend().add(left, right);
    }

    private void release() {
        for (GradientEdge edge : order) {
            if (edge == null) continue;
            BackwardNode node = edge.state().node();
            if (node != null) node.release();
        }
        gradients.clear();
    }

    private record Visit(GradientEdge edge, boolean expanded) {}
}
