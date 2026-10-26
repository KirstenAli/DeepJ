package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.function.Function;

public final class AutogradTestSupport {
    private AutogradTestSupport() {}

    public static Tensor backward(Function<Tensor, Tensor> forward, Tensor input, Tensor upstream) {
        Tensor variable = new Tensor(input).requiresGrad(true);
        forward.apply(variable).backward(upstream);
        return variable.grad();
    }

    public static void set(Tensor tensor, int row, int column, float value) {
        try (var ignored = GradMode.noGrad()) {
            tensor.set(row, column, value);
        }
    }

    public static void replaceValues(Tensor target, Tensor source) {
        Tensor.requireSameShape(target, source, "test initialization");
        try (var ignored = GradMode.noGrad()) {
            for (int row = 0; row < target.matrixRows(); row++) {
                target.setRow(row, source, row);
            }
        }
    }
}
