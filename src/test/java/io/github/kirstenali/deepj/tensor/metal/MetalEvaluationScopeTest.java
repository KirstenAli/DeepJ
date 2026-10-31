package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.testing.NdModelFixtures;
import io.github.kirstenali.deepj.training.CausalLMEvaluation;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.*;

class MetalEvaluationScopeTest extends MetalDifferentialTestSupport {
    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void evaluationPreservesMetricsAndUnrelatedCallerOutputsAcrossEveryFamilyAndDtype(String family, DType dtype) {
        Tensor.setBackend(cpu);
        var reference = NdModelFixtures.model(family, 117, new ParameterFactory(dtype));
        var expected = CausalLMEvaluation.evaluate(reference, ignored -> batch(), 8, 1);
        Tensor.setBackend(metal);
        var model = NdModelFixtures.model(family, 117, new ParameterFactory(dtype));
        Tensor caller = metal.addScalar(Tensor.ones(2, 2), 7);
        var actual = CausalLMEvaluation.evaluate(model, ignored -> batch(), 8, 1);
        double tolerance = dtype == DType.FLOAT32 ? 1e-5 : dtype == DType.FLOAT16 ? 0.002 : 0.02;
        assertEquals(expected.loss(), actual.loss(), tolerance);
        assertEquals(expected.tokens(), actual.tokens());
        assertArrayEquals(new float[]{8, 8, 8, 8}, caller.toFloatArray());
    }

    private static Batch batch() { return new Batch(new int[][]{{1, 2, 3, 4}}, new int[][]{{2, 3, 4, 5}}); }
}
