package io.github.kirstenali.deepj.chatui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GeneratedTextTest {

    @Test
    void removesPromptFromGeneratedText() {
        assertEquals(" world", GeneratedText.continuation("Hello", "Hello world"));
    }

    @Test
    void keepsTextWhenPromptWasNormalized() {
        assertEquals("Answer", GeneratedText.continuation("Question", "Answer"));
    }
}
