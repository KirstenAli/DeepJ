package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.function.UnaryOperator;

/** Activation checkpointing: run a block without saving its graph, then rebuild it during backward. */
public final class Recompute {
    private Recompute() {}

    public static Tensor forward(UnaryOperator<Tensor> block, Tensor input) {
        if (!Autograd.tracks(input)) return block.apply(input);
        Tensor output;
        try (var ignored = GradMode.noGrad()) { output = block.apply(input); }
        return Autograd.record(output, gradient -> replay(block, input, gradient), input);
    }

    private static Tensor[] replay(UnaryOperator<Tensor> block, Tensor input, Tensor gradient) {
        Tensor leaf = input.detach().requiresGrad(true);
        try (var ignored = GradMode.enabled()) {
            block.apply(leaf).backward(gradient);
        }
        return new Tensor[]{leaf.grad()};
    }
}

