package io.github.kirstenali.deepj.nn.transformer;

import io.github.kirstenali.deepj.tokenizers.Tokenizer;
import java.util.List;

public enum PromptFormat {
    PLAIN,
    INSTRUCTION_RESPONSE;

    public String apply(String prompt) {
        if (this == PLAIN || alreadyFormatted(prompt)) return prompt;
        return "Instruction:\n" + prompt.strip() + "\nResponse:";
    }

    /** Adds recent whole turns that fit in the prompt budget. */
    public String applyConversation(String prompt, List<? extends ConversationTurn> history, Tokenizer tokenizer,
                                    int window, int maxTokens) {
        if (history.isEmpty()) return apply(prompt);
        String transcript = exchange(prompt);
        String fitted = apply(prompt);
        int budget = window - Math.min(maxTokens, window / 2);
        for (int index = history.size() - 1; index >= 0; index--) {
            var turn = history.get(index);
            String candidate = exchange(turn.prompt()) + "\n" + turn.reply() + "\n\n" + transcript;
            if (tokenizer.encode(candidate).length > budget) break;
            transcript = candidate;
            fitted = candidate;
        }
        return fitted;
    }

    private String exchange(String prompt) {
        return this == PLAIN ? "User:\n" + prompt + "\nAssistant:" : apply(prompt);
    }

    private static boolean alreadyFormatted(String prompt) {
        String text = prompt.strip();
        return text.startsWith("Instruction:") && text.endsWith("Response:");
    }
}
