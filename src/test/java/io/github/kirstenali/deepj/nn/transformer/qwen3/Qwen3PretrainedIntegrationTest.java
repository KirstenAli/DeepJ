package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import java.io.IOException;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

/** Opt-in full-checkpoint comparison; requires local model files and a separately generated Transformers reference. */
@EnabledIfSystemProperty(named = "deepj.qwen3.pretrained", matches = ".+")
class Qwen3PretrainedIntegrationTest {
    @Test void completeOfficialCheckpointMatchesTransformersAcrossTwoPromptsAndCachedSteps() throws IOException {
        assertTrue(MetalBackend.isAvailable(), "This integration check requires physical Metal");
        MetalBackend metal = new MetalBackend();
        Tensor.setBackend(metal);
        try {
            compare(Qwen3Pretrained.load(Path.of(System.getProperty("deepj.qwen3.pretrained"))));
        } finally {
            metal.releaseResources();
            Tensor.setBackend(new CpuBackend());
        }
    }

    private static void compare(Qwen3 model) throws IOException {
        assertEquals(Qwen3Config.onePointSevenB(), model.config());
        assertEquals(310, model.parameters().size());
        assertTrue(model.parameters().stream().allMatch(parameter -> parameter.grad() == null));
        var reference = Qwen3Json.read(Path.of(System.getProperty("deepj.qwen3.reference")));
        try (var session = model.openGeneration()) {
            for (Object item : Qwen3Json.array(reference.get("cases"))) {
                var expected = Qwen3Json.object(item);
                int[] tokens = Qwen3Json.array(expected.get("tokens")).stream()
                        .mapToInt(value -> Math.toIntExact(Qwen3Json.integer(value))).toArray();
                float[] logits = session.apply(tokens).toFloatArray();
                assertArrayEquals(Qwen3Fixtures.floats(expected.get("logits")), logits, 0.002f);
                assertEquals(Qwen3Json.integer(expected, "next"), largest(logits));
            }
        }
    }

    private static int largest(float[] values) {
        int result = 0;
        for (int index = 1; index < values.length; index++) {
            if (values[index] > values[result]) result = index;
        }
        return result;
    }
}
