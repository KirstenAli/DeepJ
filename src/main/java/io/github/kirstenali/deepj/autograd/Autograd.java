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

    static Tensor recordBuiltin(Tensor output, BackwardRule rule, Tensor... inputs) {
        return recordBuiltinSaved(output, rule, inputs, inputs);
    }

    static Tensor recordBuiltinSaved(Tensor output, BackwardRule rule, Tensor[] inputs, Tensor... saved) {
        return attach(output, rule, inputs, saved, true);
    }

    public static Tensor recordSaved(Tensor output, BackwardRule rule, Tensor[] inputs, Tensor... saved) {
        return attach(output, rule, inputs, saved, false);
    }

    private static Tensor attach(Tensor output, BackwardRule rule, Tensor[] inputs, Tensor[] saved, boolean builtin) {
        if (!tracks(inputs)) return output;
        if (output.requiresGrad() || Arrays.asList(inputs).contains(output)) {
            throw new IllegalArgumentException("A tracked operation must return a new, untracked tensor");
        }
        output.gradientState().attach(new BackwardNode(rule, inputs, saved, builtin));
        return output;
    }

    public static void backward(Tensor output) {
        if (output.numel() != 1) {
            throw new IllegalArgumentException("Non-scalar backward requires an explicit upstream gradient");
        }
        backward(output, Tensor.from(new float[]{1.0f}, output.shape()));
    }

    public static void backward(Tensor output, Tensor gradient) {
        if (!output.requiresGrad()) throw new IllegalStateException("Tensor does not require gradients");
        Tensor.requireSameShape(output, gradient, "backward");
        try (var ignored = GradMode.noGrad()) {
            new BackwardPass(output, Tensor.backend().cast(gradient, gradient.dtype().gradientType())).run();
        }
    }
}
