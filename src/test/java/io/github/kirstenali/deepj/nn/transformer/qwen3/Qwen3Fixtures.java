package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

final class Qwen3Fixtures {
    private Qwen3Fixtures() {}

    static Path path(String dtype) {
        try { return Path.of(Qwen3Fixtures.class.getResource("/qwen3/" + dtype + "/config.json").toURI()).getParent(); }
        catch (java.net.URISyntaxException failure) { throw new IllegalStateException(failure); }
    }

    static Map<String, Object> expected(String dtype) throws IOException {
        return Qwen3Json.read(path(dtype).resolve("expected.json"));
    }

    static float[] floats(Object value) {
        List<?> array = Qwen3Json.array(value);
        float[] result = new float[array.size()];
        for (int index = 0; index < result.length; index++) {
            result[index] = (float) Qwen3Json.number(array.get(index));
        }
        return result;
    }

    static int[][] tokens(Map<String, Object> expected) {
        return Qwen3Json.array(expected.get("tokens")).stream().map(row -> Qwen3Json.array(row).stream()
                .mapToInt(value -> Math.toIntExact(Qwen3Json.integer(value))).toArray()).toArray(int[][]::new);
    }

    static float[] gradient(Qwen3Weights.Weight weight, Map<String, Object> expected) {
        float[] source = floats(Qwen3Json.object(expected.get("gradients")).get(weight.source()));
        if (!weight.transpose()) return source;
        float[] result = new float[source.length];
        for (int index = 0; index < result.length; index++) {
            result[index] = source[(index % weight.columns()) * weight.rows() + index / weight.columns()];
        }
        return result;
    }
}
