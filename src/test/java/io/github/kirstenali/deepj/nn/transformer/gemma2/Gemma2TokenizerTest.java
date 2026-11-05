package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.nn.transformer.ConversationTurn;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tokenizers.gemma.Gemma2Chat;
import io.github.kirstenali.deepj.tokenizers.gemma.GemmaTokenizer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class Gemma2TokenizerTest {
    private static final Path ROOT = Path.of("src/test/resources/gemma2");
    record Turn(String prompt, String reply) implements ConversationTurn {}
    @TempDir Path temporary;

    @Test void legacyStringMergePairsAreAccepted() throws Exception {
        java.nio.file.Files.writeString(temporary.resolve("tokenizer.json"), """
                {"model":{"type":"BPE","byte_fallback":true,
                "vocab":{"<bos>":2,"<unk>":3,"t":4,"e":5,"te":6},"merges":["t e"]},
                "added_tokens":[{"content":"<bos>","id":2}]}
                """);
        assertArrayEquals(new int[]{2, 6}, new GemmaTokenizer(temporary).encode("te"));
    }

    @Test void tokenIdsAndConversationMarkersMatchThePythonTokenizer() throws Exception {
        var tokenizer = new GemmaTokenizer(ROOT);
        var reference = Qwen3Json.read(ROOT.resolve("tokenizer-reference.json"));
        for (Object item : Qwen3Json.array(reference.get("cases"))) {
            var entry = Qwen3Json.object(item);
            int[] expected = Qwen3Json.array(entry.get("ids")).stream()
                    .mapToInt(value -> Math.toIntExact(Qwen3Json.integer(value))).toArray();
            assertArrayEquals(expected, tokenizer.encode(Qwen3Json.string(entry.get("text"))));
        }
        String formatted = Gemma2Chat.format("A fox 🦊\n雪", List.of(new Turn(" Hello ", "Hi!")));
        assertEquals(reference.get("formatted"), "<bos>" + formatted);
    }

    @Test void streamingDoesNotDisplayAnIncompleteUtf8Character() throws Exception {
        var tokenizer = new GemmaTokenizer(ROOT);
        int[] tokens = tokenizer.encode("🦊");
        assertEquals("<bos>", tokenizer.decodePartial(Arrays.copyOf(tokens, tokens.length - 1)));
        assertEquals("<bos>🦊", tokenizer.decodePartial(tokens));
        assertTrue(tokenizer.isEndOfSequence(107));
        assertFalse(tokenizer.isEndOfSequence(106));
    }
}
