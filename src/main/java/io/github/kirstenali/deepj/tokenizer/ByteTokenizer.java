package io.github.kirstenali.deepj.tokenizer;

import java.nio.charset.StandardCharsets;

/**
 * Minimal byte-level tokenizer (0-255). This is enough to train a GPT-style model end-to-end
 * without external dependencies. For real projects you can swap this for BPE/Unigram.
 */
public final class ByteTokenizer {

    public static final int VOCAB_SIZE = 256;

    public int[] encode(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            ids[i] = bytes[i] & 0xFF;
        }
        return ids;
    }

    public String decode(int[] ids) {
        byte[] bytes = new byte[ids.length];
        for (int i = 0; i < ids.length; i++) {
            int v = ids[i];
            if (v < 0 || v > 255) throw new IllegalArgumentException("Token id out of range: " + v);
            bytes[i] = (byte) v;
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }
}
