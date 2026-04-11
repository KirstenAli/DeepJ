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

    static List<Integer> mergePair(List<Integer> tokens, TokenPair pair, int newId) {
        int firstMatch = firstMatchIndex(tokens, pair);
        if (firstMatch < 0) {
            return tokens;
        }
        return applyMergeFrom(tokens, pair, newId, firstMatch);
    }

    private static int firstMatchIndex(List<Integer> tokens, TokenPair pair) {
        for (int i = 0, n = tokens.size() - 1; i < n; i++) {
            if (tokens.get(i) == pair.left() && tokens.get(i + 1) == pair.right()) {
                return i;
            }
        }
        return -1;
    }

    private static List<Integer> applyMergeFrom(List<Integer> tokens, TokenPair pair, int newId, int startFrom) {
        List<Integer> out = new ArrayList<>(tokens.size());
        out.addAll(tokens.subList(0, startFrom));
        int i = startFrom;
        int n = tokens.size();
        while (i < n) {
            if (i < n - 1 && tokens.get(i) == pair.left() && tokens.get(i + 1) == pair.right()) {
                out.add(newId);
                i += 2;
            } else {
                out.add(tokens.get(i++));
            }
        }
        return out;
    }

    static List<Integer> toTokenIds(String piece, int endOfWordId) {
        byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
        List<Integer> word = new ArrayList<>(bytes.length + 1);
        for (byte b : bytes) {
            word.add(b & 0xFF);
        }

        word.add(endOfWordId);
        return word;
    }
}