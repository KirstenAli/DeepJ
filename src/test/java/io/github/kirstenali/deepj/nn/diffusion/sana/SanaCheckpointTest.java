package io.github.kirstenali.deepj.nn.diffusion.sana;

import io.github.kirstenali.deepj.tokenizers.gemma.GemmaTokenizer;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "sana.model", matches = ".+")
class SanaCheckpointTest {
    @Test void tokenizerAndOfficialPromptMatchHuggingFace() throws Exception {
        var tokenizer = new GemmaTokenizer(Path.of(System.getProperty("sana.model"), "tokenizer"));
        var reference = Qwen3Json.read(Path.of("src/test/resources/sana/tokenizer-reference.json"));
        for (Object item : Qwen3Json.array(reference.get("cases"))) {
            var entry = Qwen3Json.object(item);
            assertArrayEquals(ids(entry.get("ids")), tokenizer.encode(Qwen3Json.string(entry.get("text"))));
        }
        var prompt = new SanaPrompt(tokenizer, Qwen3Json.string(reference.get("prompt")));
        assertArrayEquals(ids(reference.get("prompt_ids")), prompt.tokens());
        assertArrayEquals(ids(reference.get("prompt_mask")), prompt.valid());
    }

    private static int[] ids(Object values) {
        return Qwen3Json.array(values).stream().mapToInt(v -> Math.toIntExact(Qwen3Json.integer(v))).toArray();
    }
}
