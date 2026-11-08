package io.github.kirstenali.deepj.cli;

/** Presents Qwen's reasoning markers without printing partially streamed control tags. */
final class ThinkingText {
    private static final String OPEN = "<think>";
    private static final String CLOSE = "</think>";

    private ThinkingText() {}

    static String display(String text) {
        String trimmed = text.stripLeading();
        if (OPEN.startsWith(trimmed)) return "";
        if (!trimmed.startsWith(OPEN)) return text;
        int end = trimmed.indexOf(CLOSE);
        String thought = trimmed.substring(OPEN.length(), end < 0 ? trimmed.length() : end);
        if (end < 0) return "Thinking\n" + stableThought(thought);
        return "Thinking\n" + thought + "\nAnswer\n" + trimmed.substring(end + CLOSE.length());
    }

    static String completionNote(String text) {
        return text.stripLeading().startsWith(OPEN) && !text.contains(CLOSE)
                ? "\n\nThinking ended before an answer. Increase --max-tokens or turn Thinking off.\n" : "";
    }

    private static String stableThought(String text) {
        for (int size = Math.min(text.length(), CLOSE.length() - 1); size > 0; size--) {
            int start = text.length() - size;
            if (CLOSE.startsWith(text.substring(start))) return text.substring(0, start);
        }
        return text;
    }
}
