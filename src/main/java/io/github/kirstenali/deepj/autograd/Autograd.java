package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.Arrays;

public final class Autograd {
    private Autograd() {}

    public static boolean tracks(Tensor... inputs) {
        return GradMode.isEnabled() && Arrays.stream(inputs).anyMatch(Tensor::requiresGrad);
    }

    public static Tensor record(Tensor output, BackwardRule rule, Tensor... inputs) {
        return recordSaved(output, rule, inputs, inputs);
    }

    public static Tensor recordSaved(Tensor output, BackwardRule rule, Tensor[] inputs, Tensor... saved) {
        if (!tracks(inputs)) return output;
        if (Arrays.asList(inputs).contains(output)) {
            throw new IllegalArgumentException("A tracked operation must return a new tensor");
        }
        output.gradientState().attach(new BackwardNode(rule, inputs, saved));
        return output;
    }

    public static void backward(Tensor output) {
        if (output.rows != 1 || output.cols != 1) {
            throw new IllegalArgumentException("Non-scalar backward requires an explicit upstream gradient");
        }
        backward(output, Tensor.ones(1, 1));
    }

    public static void backward(Tensor output, Tensor gradient) {
        if (!output.requiresGrad()) throw new IllegalStateException("Tensor does not require gradients");
        Tensor.requireSameShape(output, gradient, "backward");
        try (var ignored = GradMode.noGrad()) {
            new BackwardPass(output, gradient).run();
        }
    }
}
