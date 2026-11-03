package io.github.kirstenali.deepj.tokenizers.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.qwen3.tokenizer", matches = ".+")
class Qwen3TokenizerGoldenTest {
    @Test void fullOfficialVocabularyMatchesHuggingFaceForUnicodeWhitespaceMergesAndSpecialTokens()
            throws IOException, URISyntaxException {
        Qwen3Tokenizer tokenizer = Qwen3Tokenizer.load(Path.of(System.getProperty("deepj.qwen3.tokenizer")));
        var file = Path.of(getClass().getResource("/qwen3/tokenizer-golden.json").toURI());
        for (Object item : Qwen3Json.array(Qwen3Json.read(file).get("cases"))) {
            var expected = Qwen3Json.object(item);
            int[] ids = Qwen3Json.array(expected.get("ids")).stream()
                    .mapToInt(value -> Math.toIntExact(Qwen3Json.integer(value))).toArray();
            String text = Qwen3Json.string(expected.get("text"));
            assertArrayEquals(ids, tokenizer.encode(text), text);
            assertEquals(expected.get("decoded"), tokenizer.decode(ids), text);
            assertEquals(expected.get("stripped"), tokenizer.decode(ids, true), text);
        }
        assertEquals(151669, tokenizer.vocabSize());
    }
}
