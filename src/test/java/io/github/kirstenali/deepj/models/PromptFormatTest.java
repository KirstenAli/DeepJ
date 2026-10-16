package io.github.kirstenali.deepj.models;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PromptFormatTest {

    @Test
    void wrapsInstructionsExactlyLikeFineTuningRecords() {
        assertEquals("Instruction:\nHello\nResponse:\n",
                PromptFormat.INSTRUCTION_RESPONSE.apply("Hello"));
    }

    @Test
    void doesNotWrapAnAlreadyFormattedPrompt() {
        String prompt = "Instruction:\nHello\nResponse:\n";
        assertEquals(prompt, PromptFormat.INSTRUCTION_RESPONSE.apply(prompt));
    }

    @Test
    void leavesCompletionPromptsUnchanged() {
        assertEquals("Once upon a time", PromptFormat.PLAIN.apply("Once upon a time"));
    }
}
