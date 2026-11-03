package io.github.kirstenali.deepj.tokenizers.qwen3;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/** Reversible byte alphabet used by Qwen's byte-level BPE files. */
final class ByteAlphabet {
    private final char[] encoded = new char[256];
    private final int[] decoded = new int[512];

    ByteAlphabet() {
        Arrays.fill(decoded, -1);
        int extra = 256;
        for (int value = 0; value < 256; value++) {
            int code = visible(value) ? value : extra++;
            encoded[value] = (char) code;
            decoded[code] = value;
        }
    }

    private static boolean visible(int value) {
        return value >= 33 && value <= 126 || value >= 161 && value <= 172 || value >= 174;
    }

    String encode(String text) {
        StringBuilder result = new StringBuilder();
        for (byte value : text.getBytes(StandardCharsets.UTF_8)) {
            result.append(encoded[Byte.toUnsignedInt(value)]);
        }
        return result.toString();
    }

    void append(String piece, ByteArrayOutputStream output) {
        for (int index = 0; index < piece.length(); index++) {
            int code = piece.charAt(index);
            if (code >= decoded.length || decoded[code] < 0) throw new IllegalArgumentException("Invalid BPE byte");
            output.write(decoded[code]);
        }
    }

    String symbol(int value) { return String.valueOf(encoded[value]); }
}
