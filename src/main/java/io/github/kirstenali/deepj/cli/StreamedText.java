package io.github.kirstenali.deepj.cli;

import java.io.PrintWriter;

/** Append cumulative updates without leaking terminal controls or incomplete UTF-8 replacement tails. */
final class StreamedText {
    private final PrintWriter output;
    private String written = "";

    StreamedText(PrintWriter output) {
        this.output = output;
    }

    void update(String text) {
        if (!Thread.currentThread().isInterrupted()) append(stable(clean(text)));
    }

    void complete(String text) { append(clean(text)); }
    boolean empty() { return written.isBlank(); }
    boolean endsWithNewline() { return written.endsWith("\n"); }

    private String clean(String text) { return CliText.safe(text); }

    private void append(String text) {
        if (!text.startsWith(written)) throw new IllegalStateException("Model revised already-streamed text");
        output.print(text.substring(written.length()));
        output.flush();
        written = text;
    }

    private static String stable(String text) {
        int end = text.length();
        while (end > 0 && text.charAt(end - 1) == '\ufffd') {
            end--;
        }
        if (end > 0 && Character.isHighSurrogate(text.charAt(end - 1))) end--;
        return text.substring(0, end);
    }
}
