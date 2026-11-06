package io.github.kirstenali.deepj.onnx;

import static org.junit.jupiter.api.Assertions.*;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Stream;

class ReferenceFixturesTest {
    static final Path FIXTURES = Path.of("src/test/resources/fixtures");

    @BeforeAll
    static void backend() {
        if (Boolean.getBoolean("onnx.test.metal")) {
            assertTrue(MetalBackend.isAvailable(), "Requested physical Metal device must be available");
            Tensor.setBackend(new MetalBackend());
        } else Tensor.setBackend(new CpuBackend());
    }

    @AfterAll
    static void release() {
        Tensor.backend().releaseResources();
        Tensor.setBackend(new CpuBackend());
    }

    @TestFactory
    Stream<DynamicTest> referenceGraphs() throws Exception {
        var manifest = Qwen3Json.read(FIXTURES.resolve("manifest.json"));
        return Qwen3Json.array(manifest.get("fixtures")).stream().map(Qwen3Json::string)
                .map(name -> DynamicTest.dynamicTest(name, () -> compare(name)));
    }

    private static void compare(String name) throws Exception {
        var reference = Qwen3Json.read(FIXTURES.resolve(name + ".json"));
        try (var scope = TensorScope.open();
                var ignored = GradMode.noGrad();
                var model = OnnxModel.load(FIXTURES.resolve(name + ".onnx"))) {
            var inputs = new LinkedHashMap<String, OnnxValue>();
            Qwen3Json.object(reference.get("inputs")).forEach((key, value) -> inputs.put(key, input(value)));
            var results = model.run(inputs);
            Qwen3Json.object(reference.get("outputs")).forEach((key, value) -> check(key, results.get(key), value));
        } finally {
            Tensor.backend().releaseTemporaryResources();
        }
    }

    static OnnxValue input(Object raw) {
        var data = Qwen3Json.object(raw);
        int[] shape = Qwen3Json.array(data.get("shape")).stream().mapToInt(v -> (int) Qwen3Json.integer(v)).toArray();
        var values = Qwen3Json.array(data.get("data"));
        int type = Qwen3Json.integer(data, "type");
        if (OnnxTensors.floating(type)) {
            DType dtype = type == 1 ? DType.FLOAT32 : type == 10 ? DType.FLOAT16 : DType.BFLOAT16;
            return OnnxValue.of(Tensor.from(floats(values), dtype, shape));
        }
        long[] integers = values.stream().mapToLong(v -> v instanceof Boolean b ? b ? 1 : 0 : Qwen3Json.integer(v))
                        .toArray();
        return OnnxValue.host(integers, shape, type);
    }

    private static float[] floats(List<?> values) {
        float[] result = new float[values.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = (float) Qwen3Json.number(values.get(i));
        }
        return result;
    }

    private static void check(String name, OnnxValue actual, Object raw) {
        OnnxValue expected = input(raw);
        assertNotNull(actual, name);
        assertEquals(expected.elementType(), actual.elementType(), name + " element type");
        assertArrayEquals(expected.shape(), actual.shape(), name + " shape");
        if (!expected.isFloating()) assertArrayEquals(expected.longs(), actual.longs(), name);
        else compareFloating(name, expected.tensor(), actual.tensor());
    }

    private static void compareFloating(String name, Tensor expected, Tensor actual) {
        float[] reference = expected.toFloatArray(), result = actual.toFloatArray();
        for (int i = 0; i < reference.length; i++) {
            float value = reference[i], tolerance = (actual.dtype() == DType.FLOAT16 ? .002f : 3e-5f)
                                    * (1 + Math.abs(value));
            assertEquals(value, result[i], tolerance, name + " element " + i);
        }
    }
}
