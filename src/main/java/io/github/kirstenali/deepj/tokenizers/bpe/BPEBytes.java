package io.github.kirstenali.deepj.tokenizers.bpe;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

final class BPEBytes {

    private BPEBytes() {
    }

    static byte[] utf8(String text) {
        return text.getBytes(StandardCharsets.UTF_8);
    }

    static String key(byte[] bytes) {
        return new String(bytes, StandardCharsets.ISO_8859_1);
    }

    static byte[] concat(byte[] a, byte[] b) {
        byte[] out = new byte[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    static List<String> splitPreserveWhitespace(String text) {
        List<String> out = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inWhitespace = false;

        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            boolean ws = Character.isWhitespace(ch);

            if (current.isEmpty()) {
                current.append(ch);
                inWhitespace = ws;
            } else if (ws == inWhitespace) {
                current.append(ch);
            } else {
                out.add(current.toString());
                current.setLength(0);
                current.append(ch);
                inWhitespace = ws;
            }
        }

        if (!current.isEmpty()) {
            out.add(current.toString());
        }

        return out;
    }
}