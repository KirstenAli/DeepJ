package io.github.kirstenali.deepj.tokenizers.bpe;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

final class BPEBytes {

    private BPEBytes() {
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
        int start = 0;
        while (start < text.length()) {
            start = appendRun(text, start, out);
        }
        return out;
    }

    private static int appendRun(String text, int start, List<String> out) {
        boolean whitespace = Character.isWhitespace(text.charAt(start));
        int end = start + 1;
        while (end < text.length() && Character.isWhitespace(text.charAt(end)) == whitespace) {
            end++;
        }
        out.add(text.substring(start, end));
        return end;
    }

    static int[] toTokenArray(String piece, int endOfWordId) {
        byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
        int[] tokens = new int[bytes.length + 1];
        for (int i = 0; i < bytes.length; i++) {
            tokens[i] = bytes[i] & 0xFF;
        }
        tokens[bytes.length] = endOfWordId;
        return tokens;
    }
}
