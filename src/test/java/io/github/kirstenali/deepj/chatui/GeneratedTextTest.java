package io.github.kirstenali.deepj.chatui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GeneratedTextTest {

    @Test
    void removesOnlyThePromptPrefix() {
        assertEquals(" world", GeneratedText.continuation("hello", "hello world"));
        assertEquals("other", GeneratedText.continuation("hello", "other"));
    }
}
