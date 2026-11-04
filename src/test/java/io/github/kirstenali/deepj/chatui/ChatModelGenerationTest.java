package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.Test;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ChatModelGenerationTest {

    @Test
    void formatsInstructionPromptsWithoutShowingThePrefix() {
        var model = new DeepJOrigin(new DeepJOriginConfig(8, 16, 8, 2, 1, 16), 1L);
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        var tokenizer = new CapturingTokenizer();
        String response = ChatModelGeneration.stream(model, tokenizer, "Hello", 0, 0.8f,
                20, 1L, text -> {});
        assertEquals("Instruction:\nHello\nResponse:", tokenizer.prompt);
        assertEquals("", response);
    }

    @Test
    void instructionConversationKeepsCompleteRecentExchangesAndReservesAnswerSpace() {
        var older = Agent.Turn.answered("Old question", "x".repeat(100));
        var recent = Agent.Turn.answered("Remember Frisky", "Okay");
        String expected = "Instruction:\nRemember Frisky\nResponse:\nOkay\n\nInstruction:\nWhat name?\nResponse:";
        var tokenizer = new ByteTokenizer();
        int window = tokenizer.encode(expected).length + 12;
        String fitted = ChatModelGeneration.conversationPrompt(PromptFormat.INSTRUCTION_RESPONSE, tokenizer,
                "What name?", List.of(older, recent), window, 12);
        assertEquals(expected, fitted);
    }

    @Test
    void plainConversationAddsRolesOnlyWhenHistoryFits() {
        var earlier = Agent.Turn.answered("Remember Frisky", "Okay");
        var tokenizer = new ByteTokenizer();
        assertEquals("User:\nRemember Frisky\nAssistant:\nOkay\n\nUser:\nWhat name?\nAssistant:",
                ChatModelGeneration.conversationPrompt(PromptFormat.PLAIN, tokenizer, "What name?",
                        List.of(earlier), 128, 12));
        assertEquals("What name?", ChatModelGeneration.conversationPrompt(PromptFormat.PLAIN,
                tokenizer, "What name?", List.of(earlier), 16, 12));
    }

    private static final class CapturingTokenizer implements Tokenizer {
        private String prompt;

        @Override
        public int[] encode(String text) {
            prompt = text;
            return new int[]{1};
        }

        @Override
        public String decode(int[] ids) {
            return prompt;
        }

        @Override
        public int vocabSize() {
            return 8;
        }
    }
}
