package io.github.kirstenali.deepj.tokenizers.gemma;

import io.github.kirstenali.deepj.nn.transformer.ConversationTurn;
import java.util.List;

/** Official user/model turn markers; the tokenizer adds the single leading BOS token. */
public final class Gemma2Chat {
    private Gemma2Chat() {}

    public static String format(String prompt, List<? extends ConversationTurn> history) {
        var text = new StringBuilder();
        for (var turn : history) {
            append(text, "user", turn.prompt());
            append(text, "model", turn.reply());
        }
        append(text, "user", prompt);
        return text.append("<start_of_turn>model\n").toString();
    }

    private static void append(StringBuilder text, String role, String content) {
        java.util.Objects.requireNonNull(content);
        text.append("<start_of_turn>").append(role).append('\n').append(content.strip())
                .append("<end_of_turn>\n");
    }
}
