package io.github.kirstenali.deepj.persistence.qwen3;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

/** Byte-order conversion around DeepJ's existing raw tensor encoding. */
final class EndianWords extends OutputStream {
    private final OutputStream output;
    private final byte[] pending;
    private int count;

    EndianWords(OutputStream output, int width) {
        this.output = output;
        pending = new byte[width];
    }

    static void reverse(byte[] bytes, int length, int width) {
        for (int start = 0; start < length; start += width) {
            for (int offset = 0; offset < width / 2; offset++) {
                byte saved = bytes[start + offset];
                bytes[start + offset] = bytes[start + width - offset - 1];
                bytes[start + width - offset - 1] = saved;
            }
        }
    }

    @Override public void write(int value) throws IOException {
        pending[count++] = (byte) value;
        if (count == pending.length) {
            reverse(pending, count, count);
            output.write(pending);
            count = 0;
        }
    }

    @Override public void write(byte[] bytes, int offset, int length) throws IOException {
        if (count != 0 || length % pending.length != 0) {
            for (int index = offset; index < offset + length; index++) {
                write(bytes[index]);
            }
            return;
        }
        byte[] swapped = Arrays.copyOfRange(bytes, offset, offset + length);
        reverse(swapped, length, pending.length);
        output.write(swapped);
    }
}
