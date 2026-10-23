package io.github.kirstenali.deepj.autograd;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertFalse;

class AttentionGradientTest {
    private TensorBackend previous;

    @BeforeEach
    void useCpu() {
        previous = Tensor.backend();
        Tensor.setBackend(new CpuBackend());
    }

    @AfterEach
    void restoreBackend() { Tensor.setBackend(previous); }

    @Test
    void fallbackDifferentiatesQueryKeyAndValueAgainstFiniteDifferences() {
        Tensor query = point(), key = point().multiplyScalar(0.7f), value = point().addScalar(0.3f);
        OperationGradientAssertions.check(x -> AttentionOps.causal(x, key, value, 2, 0.5f), query);
        OperationGradientAssertions.check(x -> AttentionOps.causal(query, x, value, 2, 0.5f), key);
        OperationGradientAssertions.check(x -> AttentionOps.causal(query, key, x, 2, 0.5f), value);
    }

    @Test
    void inferenceFallbackDoesNotBuildAnAutogradGraph() {
        Tensor input = point().requiresGrad(true);
        try (var ignored = GradMode.noGrad()) {
            assertFalse(AttentionOps.causal(input, input, input, 2, 0.5f).requiresGrad());
        }
    }

    private static Tensor point() { return Tensor.random(6, 4, new Random(173)); }
}
