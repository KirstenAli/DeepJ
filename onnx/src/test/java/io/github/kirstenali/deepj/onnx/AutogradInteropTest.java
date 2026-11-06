package io.github.kirstenali.deepj.onnx;

import static org.junit.jupiter.api.Assertions.*;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class AutogradInteropTest {
    @BeforeAll
    static void backend() {
        ReferenceFixturesTest.backend();
    }

    @AfterAll
    static void release() {
        ReferenceFixturesTest.release();
    }

    @ParameterizedTest
    @ValueSource(strings = {"mlp", "conv2d", "gru_forward", "transformer"})
    void gradientsMatchIndependentFiniteDifferences(String name) throws Exception {
        var fixture = Qwen3Json.read(ReferenceFixturesTest.FIXTURES.resolve(name + ".json"));
        var oracle = Qwen3Json.read(ReferenceFixturesTest.FIXTURES.resolve(name + ".grad.json"));
        try (var scope = TensorScope.open();
                var model =
                        OnnxModel.load(ReferenceFixturesTest.FIXTURES.resolve(name + ".onnx"))) {
            var parameters =
                    model.trainable(
                            Qwen3Json.array(oracle.get("parameters")).toArray(String[]::new));
            var inputs = differentiableInputs(fixture);
            Tensor y = model.run(inputs).get("Y").tensor();
            y.multiply(y).sumTensor().backward();
            compareGradients(parameters, inputs, oracle);
            assertReusedParameters(model, parameters);
        } finally {
            Tensor.backend().releaseTemporaryResources();
        }
    }

    private static void check(String name, Tensor actual, Object raw) {
        assertNotNull(actual, name + " gradient");
        Tensor expected = ReferenceFixturesTest.input(raw).tensor();
        assertArrayEquals(expected.shape(), actual.shape());
        float[] values = expected.toFloatArray(), result = actual.toFloatArray();
        for (int i = 0; i < values.length; i++) {
            assertEquals(
                    values[i],
                    result[i],
                    .001f + Math.abs(values[i]) * .01f,
                    name + " gradient " + i);
        }
    }

    private static void compareGradients(
            java.util.List<Parameter> parameters,
            Map<String, OnnxValue> inputs,
            Map<String, Object> oracle) {
        var gradients = new LinkedHashMap<String, Tensor>();
        parameters.forEach(p -> gradients.put(p.name(), p.grad()));
        gradients.put("X", inputs.get("X").tensor().grad());
        Qwen3Json.object(oracle.get("gradients"))
                .forEach((key, value) -> check(key, gradients.get(key), value));
    }

    private static Map<String, OnnxValue> differentiableInputs(Map<String, Object> fixture) {
        var inputs = new LinkedHashMap<String, OnnxValue>();
        Qwen3Json.object(fixture.get("inputs"))
                .forEach((key, value) -> inputs.put(key, ReferenceFixturesTest.input(value)));
        inputs.get("X").tensor().requiresGrad(true);
        return inputs;
    }

    private static void assertReusedParameters(
            OnnxModel model, java.util.List<Parameter> parameters) throws Exception {
        assertEquals(
                parameters,
                model.trainable(parameters.get(0).name()),
                "Shared weights must not be registered twice");
    }
}
