package io.github.kirstenali.deepj.testing;

import io.github.kirstenali.deepj.tensor.Tensor;

import static io.github.kirstenali.deepj.testing.NumericalGradientAssertions.assertTensorGradients;

/** Identical independent finite-difference controls run against the selected CPU or GPU backend. */
public final class NdStatisticsGradientChecks {
    private NdStatisticsGradientChecks() {}

    public static void assertMeanAndVariance() {
        Tensor input = Tensor.from(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 2, 1, 3);
        assertTensorGradients(input, value -> value.meanAxes(0, 2), Tensor.from(new float[]{0.7f}, 1),
                1e-3f, 1e-4f, 1e-3f);
        assertTensorGradients(input, value -> value.varianceAxes(new int[]{0, -1}, true, 1),
                Tensor.from(new float[]{0.7f}, 1, 1, 1), 1e-3f, 1e-4f, 1e-3f);
    }
}
