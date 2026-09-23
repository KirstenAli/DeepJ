package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.layers.Layer;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public final class NumericalGradientAssertions {

    private NumericalGradientAssertions() {}

    public static void assertLayerGradients(Layer layer, Tensor input, Tensor upstream,
                                            float epsilon, float absoluteTolerance,
                                            float relativeTolerance) {
        layer.zeroGrad();
        Tensor output = layer.forward(input);
        requireSameShape(output, upstream);
        Tensor analyticInput = new Tensor(layer.backward(upstream));
        List<Tensor> analyticParameters = copyGradients(layer.parameters());
        assertNumericalGradient(input, analyticInput, () -> layer.forward(input), upstream,
                epsilon, absoluteTolerance, relativeTolerance, "input");
        assertParameterGradients(layer.parameters(), analyticParameters, () -> layer.forward(input),
                upstream, epsilon, absoluteTolerance, relativeTolerance);
    }

    public static void assertParameterGradients(List<Parameter> parameters, Supplier<Tensor> forward,
                                                Tensor upstream, float epsilon,
                                                float absoluteTolerance, float relativeTolerance) {
        assertParameterGradients(parameters, copyGradients(parameters), forward, upstream,
                epsilon, absoluteTolerance, relativeTolerance);
    }

    private static void assertParameterGradients(List<Parameter> parameters, List<Tensor> analytic,
                                                 Supplier<Tensor> forward, Tensor upstream, float epsilon,
                                                 float absoluteTolerance, float relativeTolerance) {
        for (int i = 0; i < parameters.size(); i++) {
            assertNumericalGradient(parameters.get(i).value, analytic.get(i), forward, upstream,
                    epsilon, absoluteTolerance, relativeTolerance, "parameter " + i);
        }
    }

    private static void assertNumericalGradient(Tensor variable, Tensor analytic, Supplier<Tensor> forward,
                                                Tensor upstream, float epsilon, float absoluteTolerance,
                                                float relativeTolerance, String label) {
        for (int i = 0; i < variable.data.length; i++) {
            float numerical = centralDifference(variable, i, forward, upstream, epsilon);
            assertClose(analytic.data[i], numerical, absoluteTolerance, relativeTolerance,
                    label + " gradient at flat index " + i);
        }
    }

    private static float centralDifference(Tensor variable, int index, Supplier<Tensor> forward,
                                           Tensor upstream, float epsilon) {
        float original = variable.data[index];
        float step = epsilon * Math.max(1.0f, Math.abs(original));
        variable.data[index] = original + step;
        float plus = objective(forward.get(), upstream);
        variable.data[index] = original - step;
        float minus = objective(forward.get(), upstream);
        variable.data[index] = original;
        return (plus - minus) / (2.0f * step);
    }

    private static float objective(Tensor output, Tensor upstream) {
        requireSameShape(output, upstream);
        output.materialize();
        upstream.materialize();
        float result = 0.0f;
        for (int i = 0; i < output.data.length; i++) {
            result += output.data[i] * upstream.data[i];
        }
        return result;
    }

    private static List<Tensor> copyGradients(List<Parameter> parameters) {
        List<Tensor> gradients = new ArrayList<>(parameters.size());
        for (Parameter parameter : parameters) {
            gradients.add(new Tensor(parameter.grad));
        }
        return gradients;
    }

    private static void requireSameShape(Tensor expected, Tensor actual) {
        assertEquals(expected.rows, actual.rows, "row count");
        assertEquals(expected.cols, actual.cols, "column count");
    }

    private static void assertClose(float analytic, float numerical, float absoluteTolerance,
                                    float relativeTolerance, String label) {
        float tolerance = absoluteTolerance
                + relativeTolerance * Math.max(Math.abs(analytic), Math.abs(numerical));
        assertTrue(Float.isFinite(analytic), label + " analytic value is not finite: " + analytic);
        assertTrue(Float.isFinite(numerical), label + " numerical value is not finite: " + numerical);
        assertTrue(Math.abs(analytic - numerical) <= tolerance,
                label + " analytic=" + analytic + " numerical=" + numerical + " tolerance=" + tolerance);
    }
}
