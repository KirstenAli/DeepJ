package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.layers.transformer.attention.MultiHeadSelfAttention;
import io.github.kirstenali.deepj.models.DecoderOnlyModel;
import io.github.kirstenali.deepj.models.nova.*;
import io.github.kirstenali.deepj.models.orbit.*;
import io.github.kirstenali.deepj.models.origin.*;
import io.github.kirstenali.deepj.models.prism.*;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class MetalSharedAttentionModelTest extends MetalDifferentialTestSupport {
    private enum Kind { ORIGIN, ORBIT, PRISM, NOVA }
    private record Step(float loss, List<Tensor> gradients, List<Tensor> weights) {}
    private final int[] calls = new int[3];

    @BeforeEach
    void requireBlockAttention() { assumeTrue(MetalNative.TILED_ATTENTION_AVAILABLE); }

    @ParameterizedTest
    @EnumSource(Kind.class)
    void everyArchitectureRoutesToSharedAttentionAndMatchesUnfusedTraining(Kind kind) {
        List<Step> expected = train(kind, backend(false));
        List<Step> actual = train(kind, backend(true));
        for (int step = 0; step < expected.size(); step++) {
            compare(expected.get(step), actual.get(step));
        }
        assertEquals(6, calls[0], "two attention layers per update");
        assertEquals(6, calls[1], "two attention backward operations per update");
    }

    @ParameterizedTest
    @EnumSource(Kind.class)
    void sharedPathDoesNotDownloadIntermediateTensors(Kind kind) {
        Tensor.setBackend(backend(true));
        DecoderOnlyModel model = model(kind);
        Tensor output = model.forward(batch().x()[0]);
        output.backward(random(output.rows, output.cols, 203));
        assertEquals(2, calls[0]);
        assertEquals(2, calls[1]);
        assertEquals(0, calls[2], "intermediate downloads");
    }

    @Test
    void nonCausalAttentionKeepsTheUnfusedPath() {
        Tensor.setBackend(backend(true));
        var layer = new MultiHeadSelfAttention(128, 2, false, new Random(211));
        Tensor output = layer.forward(random(256, 128, 223));
        output.backward(random(256, 128, 227));
        assertEquals(0, calls[0]);
        assertEquals(0, calls[1]);
        assertEquals(0, calls[2]);
    }

    private TensorBackend backend(boolean fused) {
        Class<?>[] interfaces = fused ? new Class<?>[]{TensorBackend.class, CausalAttentionBackend.class}
                : new Class<?>[]{TensorBackend.class};
        return (TensorBackend) Proxy.newProxyInstance(TensorBackend.class.getClassLoader(), interfaces,
                (proxy, method, arguments) -> invoke(method, arguments));
    }

    private Object invoke(Method method, Object[] arguments) throws Throwable {
        switch (method.getName()) {
            case "causalAttention" -> calls[0]++;
            case "causalAttentionBackward" -> calls[1]++;
            case "materializeTensor" -> calls[2]++;
            default -> { }
        }
        try { return method.invoke(metal, arguments); }
        catch (InvocationTargetException error) { throw error.getCause(); }
    }

    private static DecoderOnlyModel model(Kind kind) {
        return switch (kind) {
            case ORIGIN -> new DeepJOrigin(new DeepJOriginConfig(31, 256, 128, 2, 2, 256), 197);
            case ORBIT -> new DeepJOrbit(new DeepJOrbitConfig(31, 256, 128, 2, 2, 256), 197);
            case PRISM -> new DeepJPrism(new DeepJPrismConfig(31, 256, 128, 2, 2, 256, 64, 32), 197);
            case NOVA -> new DeepJNova(new DeepJNovaConfig(31, 256, 128, 2, 1, 2, 256), 197);
        };
    }

    private static List<Step> train(Kind kind, TensorBackend backend) {
        Tensor.setBackend(backend);
        DecoderOnlyModel model = model(kind);
        var trainer = CausalLMTraining.trainer(model, ignored -> batch(), 1e-4f);
        List<Step> result = new ArrayList<>();
        for (int step = 0; step < 3; step++) {
            float loss = trainer.trainStep(1);
            result.add(snapshot(model, loss));
            backend.releaseTemporaryResources();
        }
        return result;
    }

    private static Step snapshot(DecoderOnlyModel model, float loss) {
        List<Tensor> weights = model.parameters().stream().map(parameter -> new Tensor(parameter.value)).toList();
        return new Step(loss, copyGradients(model.parameters()), weights);
    }

    private static void compare(Step expected, Step actual) {
        assertEquals(expected.loss(), actual.loss(), 2e-5f + Math.abs(expected.loss()) * 1e-5f);
        assertGradientListsClose(expected.gradients(), actual.gradients(), 2e-6f, 2e-3f);
        assertGradientListsClose(expected.weights(), actual.weights(), 2e-6f, 2e-4f);
    }

    private static Batch batch() {
        int[][] input = new int[1][256], target = new int[1][256];
        for (int index = 0; index < 256; index++) {
            input[0][index] = index % 31;
            target[0][index] = (index + 1) % 31;
        }
        return new Batch(input, target);
    }
}
