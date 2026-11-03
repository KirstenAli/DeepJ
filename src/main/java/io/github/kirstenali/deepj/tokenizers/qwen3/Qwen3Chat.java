package io.github.kirstenali.deepj.tokenizers.qwen3;

import java.util.List;
import java.util.Objects;

/** Qwen3 text conversation formatting, including thinking and non-thinking generation prompts. */
public final class Qwen3Chat {
    private Qwen3Chat() {}
    public enum Role { SYSTEM, USER, ASSISTANT }

    public record Message(Role role, String content, String reasoningContent) {
        public Message(Role role, String content) { this(role, content, null); }
        public Message {
            Objects.requireNonNull(role, "role");
            Objects.requireNonNull(content, "content");
            if (role != Role.ASSISTANT && reasoningContent != null) {
                throw new IllegalArgumentException("Only assistant messages may contain reasoning");
            }
        }
    }

    public static String format(List<Message> messages, boolean addGenerationPrompt, boolean enableThinking) {
        if (messages.isEmpty()) throw new IllegalArgumentException("A conversation must contain a message");
        StringBuilder result = new StringBuilder();
        int lastUser = lastUser(messages);
        for (int index = 0; index < messages.size(); index++) {
            Message message = messages.get(index);
            String content = message.role() == Role.ASSISTANT
                    ? assistant(message, index > lastUser, index == messages.size() - 1) : message.content();
            result.append("<|im_start|>").append(message.role().name().toLowerCase(java.util.Locale.ROOT))
                    .append('\n').append(content).append("<|im_end|>\n");
        }
        if (addGenerationPrompt) {
            result.append("<|im_start|>assistant\n");
            if (!enableThinking) result.append("<think>\n\n</think>\n\n");
        }
        return result.toString();
    }

    private static int lastUser(List<Message> messages) {
        for (int index = messages.size() - 1; index >= 0; index--) {
            Message message = messages.get(index);
            if (message.role() == Role.USER && !toolResponse(message.content())) return index;
        }
        return messages.size() - 1;
    }

    private static boolean toolResponse(String text) {
        return text.startsWith("<tool_response>") && text.endsWith("</tool_response>");
    }

    private static String assistant(Message message, boolean currentTurn, boolean last) {
        String content = message.content(), reasoning = message.reasoningContent();
        if (reasoning == null && content.contains("</think>")) {
            int end = content.indexOf("</think>");
            reasoning = trimNewlines(content.substring(0, end).replaceFirst("(?s)^.*<think>", ""));
            content = content.substring(content.lastIndexOf("</think>") + 8).replaceFirst("^\\n+", "");
        }
        if (currentTurn && (last || reasoning != null && !reasoning.isEmpty())) {
            return "<think>\n" + trimNewlines(reasoning == null ? "" : reasoning)
                    + "\n</think>\n\n" + content.replaceFirst("^\\n+", "");
        }
        return content;
    }

    private static String trimNewlines(String text) { return text.replaceAll("^\\n+|\\n+$", ""); }
}
