package io.github.kirstenali.deepj.tokenizers.qwen3;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3TokenizerTest {
    @TempDir Path directory;

    @Test void byteBpeAndAddedTokensFollowRanksWithoutAddingBosOrWhitespace() throws IOException {
        Qwen3Tokenizer tokenizer = tokenizer();
        assertArrayEquals(new int[]{257}, tokenizer.encode("abc"));
        assertArrayEquals(new int[0], tokenizer.encode(""));
        String text = "👩🏽‍💻 café\t你好\r\n 1234567890";
        assertEquals(text, tokenizer.decode(tokenizer.encode(text)));
        assertEquals("é", tokenizer.decode(tokenizer.encode("e\u0301")));
        assertArrayEquals(new int[]{259, 257, 260}, tokenizer.encode("<|im_start|>abc<|im_end|>"));
        assertEquals("abc<think>", tokenizer.decode(new int[]{259, 257, 260, 261}, true));
        assertTrue(tokenizer.isEndOfSequence(258));
        assertTrue(tokenizer.isEndOfSequence(260));
        assertFalse(tokenizer.isEndOfSequence(259));
        assertThrows(IllegalArgumentException.class, () -> tokenizer.decode(new int[]{999}));
    }

    @Test void invalidOrUnsupportedTokenizerOptionsFailDuringLoading() throws IOException {
        var root = Qwen3TokenizerFixture.recipe();
        Qwen3Json.object(root.get("model")).put("unk_token", "unknown");
        Files.writeString(directory.resolve("tokenizer.json"), Qwen3Json.stringify(root));
        assertThrows(IOException.class, () -> Qwen3Tokenizer.load(directory));
    }

    private Qwen3Tokenizer tokenizer() throws IOException {
        Qwen3TokenizerFixture.write(directory);
        return Qwen3Tokenizer.load(directory);
    }

}
