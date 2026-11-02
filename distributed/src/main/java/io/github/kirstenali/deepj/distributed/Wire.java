package io.github.kirstenali.deepj.distributed;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

final class Wire {
    static final int MAGIC = 0x444A4450;
    static final int VERSION = 1;
    static final int MAX_PAYLOAD = 16 * 1024 * 1024;
    static final int SUM = 1, BROADCAST = 2, GATHER = 3;
    private Wire() {}

    @FunctionalInterface interface Writer { void write(DataOutputStream output) throws IOException; }

    static byte[] encode(Writer writer) {
        var bytes = new ByteArrayOutputStream();
        try (var output = new DataOutputStream(bytes)) { writer.write(output); }
        catch (IOException failure) { throw new UncheckedIOException(failure); }
        return bytes.toByteArray();
    }

    static DataInputStream input(byte[] bytes) { return new DataInputStream(new ByteArrayInputStream(bytes)); }

    static void writeBytes(DataOutputStream output, byte[] bytes) throws IOException {
        checkLength(bytes.length);
        output.writeInt(bytes.length);
        output.write(bytes);
    }

    static byte[] readBytes(DataInputStream input) throws IOException {
        int length = input.readInt();
        checkLength(length);
        byte[] bytes = new byte[length];
        input.readFully(bytes);
        return bytes;
    }

    static void checkLength(int length) throws IOException {
        if (length < 0 || length > MAX_PAYLOAD) throw new IOException("Invalid collective payload length");
    }
}
