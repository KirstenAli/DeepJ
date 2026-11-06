package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tokenizers.gemma.GemmaTokenizer;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "sanavideo.model", matches = ".+")
class SanaVideoCheckpointTest {
    private static final Path MODEL = Path.of(System.getProperty("sanavideo.model", "."));

    @Test void officialTokenizerMatchesDiffusersPositiveAndUnconditionalPrompts() throws Exception {
        var reference = Qwen3Json.read(Path.of("src/test/resources/sana-video/tokenizer-reference.json"));
        var tokenizer = new GemmaTokenizer(MODEL.resolve("tokenizer"));
        for (Object value : Qwen3Json.array(reference.get("cases"))) {
            var entry = Qwen3Json.object(value);
            var prompt = new SanaVideoPrompt(tokenizer, Qwen3Json.string(entry.get("text")),
                    Boolean.TRUE.equals(entry.get("positive")));
            assertArrayEquals(ints(entry.get("ids")), prompt.tokens());
            assertArrayEquals(ints(entry.get("valid")), prompt.valid());
            assertArrayEquals(ints(entry.get("selected")), prompt.selected());
            assertArrayEquals(ints(entry.get("mask")), prompt.mask());
        }
    }

    @Test void realCheckpointMetadataIncludesVideoPatchAndCausalDecoderWeights() throws Exception {
        SanaVideoSnapshot.validate(MODEL);
        try (var video = TensorWeights.open(MODEL.resolve("transformer"), DType.FLOAT32);
             var vae = TensorWeights.open(MODEL.resolve("vae"), DType.FLOAT32)) {
            assertArrayEquals(new int[]{2240, 16, 1, 2, 2},
                    video.descriptors().get("patch_embedding.weight").shape());
            assertArrayEquals(new int[]{384, 16, 3, 3, 3},
                    vae.descriptors().get("decoder.conv_in.weight").shape());
            assertTrue(video.parameters().isEmpty());
            assertTrue(vae.parameters().isEmpty());
        }
    }

    private static int[] ints(Object array) {
        return Qwen3Json.array(array).stream().mapToInt(value -> Math.toIntExact(Qwen3Json.integer(value))).toArray();
    }
}
