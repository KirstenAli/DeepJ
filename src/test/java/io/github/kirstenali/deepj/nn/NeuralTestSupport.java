package io.github.kirstenali.deepj.nn;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

final class NeuralTestSupport {
    private NeuralTestSupport() {}

    static Tensor values(int... shape) {
        int count = 1;
        for (int size : shape) { count *= size; }
        float[] values = new float[count];
        for (int index = 0; index < count; index++) { values[index] = (index % 11 - 5) / 13.0f; }
        return Tensor.from(values, shape);
    }

    static void initialize(Layer model) {
        int index = 0;
        try (var ignored = GradMode.noGrad()) {
            for (Parameter parameter : model.parameters()) {
                parameter.value.copyFrom(values(parameter.value.shape()).addScalar(index++ / 31.0f));
            }
        }
    }

    static void gradients(Layer model, Tensor input, Tensor... states) {
        input.requiresGrad(true);
        model.zeroGrad();
        Tensor output = model.forward(input), upstream = values(output.shape()).addScalar(0.17f);
        output.backward(upstream);
        List<Tensor> variables = new ArrayList<>(List.of(input));
        for (Parameter parameter : model.parameters()) { variables.add(parameter.value); }
        variables.addAll(List.of(states));
        List<float[]> analytical = variables.stream().map(variable -> variable.grad().toFloatArray()).toList();
        for (int index = 0; index < variables.size(); index++) {
            finiteDifference(model, input, upstream, variables.get(index), analytical.get(index));
        }
    }

    private static void finiteDifference(Layer model, Tensor input, Tensor upstream,
                                         Tensor variable, float[] expected) {
        float[] original = variable.toFloatArray();
        for (int index = 0; index < original.length; index++) {
            set(variable, index, original[index] + 0.001f);
            float plus = objective(model, input, upstream);
            set(variable, index, original[index] - 0.001f);
            float minus = objective(model, input, upstream);
            set(variable, index, original[index]);
            assertEquals((plus - minus) / 0.002f, expected[index], 0.0025f,
                    "finite difference at element " + index + " of " + java.util.Arrays.toString(variable.shape()));
        }
    }

    private static float objective(Layer model, Tensor input, Tensor upstream) {
        try (var ignored = GradMode.noGrad(); var scope = TensorScope.open()) {
            return model.forward(input).multiply(upstream).sumTensor().item();
        }
    }

    static void set(Tensor value, int index, float replacement) {
        int[] coordinate = new int[value.rank()];
        for (int axis = coordinate.length - 1; axis >= 0; axis--) {
            coordinate[axis] = index % value.size(axis);
            index /= value.size(axis);
        }
        try (var ignored = GradMode.noGrad()) { value.setAt(replacement, coordinate); }
    }
}
