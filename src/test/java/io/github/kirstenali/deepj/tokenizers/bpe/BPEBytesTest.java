package io.github.kirstenali.deepj.tokenizers.bpe;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BPEBytesTest {

    @Test
    void splitPreserveWhitespace_keepsRunsTogether() {
        List<String> parts = BPEBytes.splitPreserveWhitespace("hello   world\t!\n");
        assertEquals(List.of("hello", "   ", "world", "\t", "!", "\n"), parts);
    }

    @Test
    void splitPreserveWhitespace_handlesEmptyString() {
        List<String> parts = BPEBytes.splitPreserveWhitespace("");
        assertEquals(List.of(), parts);
    }

    @Test
    void splitPreserveWhitespace_handlesOnlyWhitespace() {
        List<String> parts = BPEBytes.splitPreserveWhitespace("   \t\n");
        assertEquals(List.of("   \t\n"), parts);
    }
}