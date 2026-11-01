package io.github.kirstenali.deepj.tensor;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.nn.transformer.CausalLM;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.CausalLMEvaluation;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class EvaluationScopeTest {
    private final TensorBackend previous = Tensor.backend();

    @AfterEach void restore() { Tensor.setBackend(previous); }

    @Test
    void libraryEvaluationClosesPrivateViewsPerSequenceWithoutRelyingOnCollection() {
        List<Tensor> matrices = new ArrayList<>();
        int[] releases = new int[2];
        Tensor.setBackend(tracking(matrices, releases));
        var model = new DeepJOrigin(new DeepJOriginConfig(8, 4, 4, 2, 1, 8), 2L);
        var result = CausalLMEvaluation.evaluate(model, ignored -> batch(), 3, 1);
        assertTrue(Double.isFinite(result.loss()));
        assertEquals(0, releases[0], "Evaluation must not discard unrelated public temporary tensors");
        assertEquals(1, releases[1]);
        assertFalse(matrices.isEmpty());
        assertTrue(matrices.stream().allMatch(tensor -> tensor.memory().canReleaseTemporary()));
    }

    @Test
    void callerModelViewsStayPublicEvenWhenEvaluationStartsInsideAScope() {
        Tensor.setBackend(new CpuBackend());
        var saved = new AtomicReference<Tensor>();
        try (var ignored = TensorScope.open()) {
            CausalLMEvaluation.evaluate(custom(saved), size -> batch(), 2, 1);
        }
        assertFalse(saved.get().memory().canReleaseTemporary());
    }

    private static CausalLM custom(AtomicReference<Tensor> saved) {
        return new CausalLM() {
            @Override public Tensor forward(int[] ids) {
                assertNull(TensorScope.current());
                Tensor logits = Tensor.aliasStorage(Tensor.ones(ids.length, 8));
                saved.set(logits);
                return logits;
            }
            @Override public float gradClipNorm() { return 1; }
            @Override public List<Parameter> parameters() { return List.of(); }
        };
    }

    private static TensorBackend tracking(List<Tensor> matrices, int[] releases) {
        var cpu = new CpuBackend();
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(),
                new Class<?>[]{TensorBackend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("releaseTemporaryResources")) releases[0]++;
                    if (method.getName().equals("releaseResources")) releases[1]++;
                    try {
                        Object result = method.invoke(cpu, args);
                        if (method.getName().equals("matmul")) matrices.add((Tensor) result);
                        return result;
                    } catch (InvocationTargetException failure) { throw failure.getCause(); }
                });
    }

    private static Batch batch() { return new Batch(new int[][]{{1, 2}}, new int[][]{{2, 3}}); }
}
