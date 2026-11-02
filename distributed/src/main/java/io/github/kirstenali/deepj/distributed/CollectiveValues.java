package io.github.kirstenali.deepj.distributed;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

final class CollectiveValues {
    private CollectiveValues() {}

    static byte[] floats(float[] values) {
        if (values.length > Wire.MAX_PAYLOAD / Float.BYTES) {
            throw new IllegalArgumentException("Collective exceeds maximum payload size");
        }
        var buffer = ByteBuffer.allocate(values.length * Float.BYTES);
        for (float value : values) { buffer.putFloat(value); }
        return buffer.array();
    }

    static void copy(byte[] bytes, float[] target) {
        if (bytes.length != target.length * Float.BYTES) throw new CollectiveException("Float payload mismatch");
        var buffer = ByteBuffer.wrap(bytes);
        for (int index = 0; index < target.length; index++) { target[index] = buffer.getFloat(); }
    }

    static byte[] sum(List<byte[]> values) throws IOException {
        int length = values.get(0).length;
        if (length % Float.BYTES != 0) throw new IOException("Invalid float payload");
        float[] result = new float[length / Float.BYTES];
        for (byte[] bytes : values) { add(result, bytes); }
        return floats(result);
    }

    private static void add(float[] result, byte[] bytes) throws IOException {
        if (bytes.length != result.length * Float.BYTES) throw new IOException("Reduction sizes differ");
        var buffer = ByteBuffer.wrap(bytes);
        for (int index = 0; index < result.length; index++) { result[index] += buffer.getFloat(); }
    }

    static byte[] gathered(List<byte[]> values) {
        long length = Integer.BYTES;
        for (byte[] value : values) { length += Integer.BYTES + value.length; }
        if (length > Wire.MAX_PAYLOAD) throw new CollectiveException("Gathered payload exceeds frame limit");
        return Wire.encode(output -> {
            output.writeInt(values.size());
            for (byte[] value : values) { Wire.writeBytes(output, value); }
        });
    }

    static List<byte[]> unpack(byte[] bytes, int worldSize) throws IOException {
        try (var input = Wire.input(bytes)) {
            if (input.readInt() != worldSize) throw new IOException("Gather worker count mismatch");
            List<byte[]> result = new ArrayList<>();
            for (int rank = 0; rank < worldSize; rank++) { result.add(Wire.readBytes(input)); }
            if (input.read() != -1) throw new IOException("Trailing gather data");
            return List.copyOf(result);
        }
    }
}
