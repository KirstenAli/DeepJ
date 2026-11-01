package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import org.junit.jupiter.api.Test;

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
