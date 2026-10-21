package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;

public final class OperationGradientAssertions {
    private OperationGradientAssertions() {}

    public static void check(Function<Tensor, Tensor> forward, Tensor point) {
        Tensor variable = new Tensor(point).requiresGrad(true);
        Tensor output = forward.apply(variable);
        output.backward(Tensor.ones(output.rows, output.cols));
        Tensor analytic = variable.grad();
        analytic.materialize();
        try (var ignored = GradMode.noGrad()) {
            for (int index = 0; index < point.data.length; index++) {
                float numerical = difference(forward, point, index);
                assertEquals(numerical, analytic.data[index], 3e-3f + Math.abs(numerical) * 2e-2f,
                        "gradient at flat index " + index);
            }
        }
    }

    private static float difference(Function<Tensor, Tensor> forward, Tensor point, int index) {
        int row = index / point.cols, column = index % point.cols;
        float original = point.get(row, column), epsilon = 1e-3f;
        try {
            point.set(row, column, original + epsilon);
            float plus = forward.apply(point).sum();
            point.set(row, column, original - epsilon);
            return (plus - forward.apply(point).sum()) / (2 * epsilon);
        } finally {
            point.set(row, column, original);
        }
    }
}
