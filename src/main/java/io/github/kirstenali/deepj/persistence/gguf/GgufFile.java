package io.github.kirstenali.deepj.persistence.gguf;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/** Metadata and validated byte ranges for dense F32 and block Q8_0 GGUF tensors. */
public record GgufFile(Path path, Map<String, Object> metadata, Map<String, Entry> tensors, long size) {
    public static final int F32 = 0;
    public static final int Q8_0 = 8;

    public record Entry(int rows, int width, int type, long offset, long bytes) {
        Entry at(long data) { return new Entry(rows, width, type, Math.addExact(data, offset), bytes); }
    }

    public static GgufFile inspect(Path path) throws IOException {
        try (var file = new RandomAccessFile(path.toFile(), "r")) {
            var input = new GgufInput(file);
            if (file.readInt() != 0x47475546 || input.i32() != 3) throw new IOException("Expected GGUF v3");
            int tensors = input.count(100_000), metadata = input.count(100_000);
            Map<String, Object> fields = fields(input, metadata);
            Map<String, Entry> entries = entries(input, tensors);
            int alignment = alignment(fields);
            long data = aligned(file.getFilePointer(), alignment), size = file.length();
            return new GgufFile(path, Map.copyOf(fields), validated(entries, data, size, alignment), size);
        } catch (ArithmeticException invalid) {
            throw new IOException("Invalid GGUF tensor size or offset", invalid);
        }
    }

    private static Map<String, Object> fields(GgufInput input, int count) throws IOException {
        Map<String, Object> fields = new HashMap<>();
        var names = new HashSet<String>();
        for (int index = 0; index < count; index++) {
            String key = input.string();
            if (!names.add(key)) throw new IOException("Duplicate GGUF metadata: " + key);
            Object value = input.metadata(input.i32());
            if (value != null) fields.put(key, value);
        }
        return fields;
    }

    private static Map<String, Entry> entries(GgufInput input, int count) throws IOException {
        Map<String, Entry> result = new HashMap<>();
        for (int index = 0; index < count; index++) {
            String name = input.string();
            if (result.putIfAbsent(name, entry(input)) != null) throw new IOException("Duplicate GGUF tensor: " + name);
        }
        return result;
    }

    private static Entry entry(GgufInput input) throws IOException {
        int rank = input.i32();
        if (rank < 1 || rank > 2) throw new IOException("Expected GGUF vector or matrix");
        int width = input.count(Integer.MAX_VALUE), rows = rank == 1 ? 1 : input.count(Integer.MAX_VALUE);
        if (width == 0 || rows == 0) throw new IOException("Empty GGUF tensor");
        int type = input.i32();
        return new Entry(rows, width, type, input.i64(), bytes(rows, width, type));
    }

    private static long bytes(int rows, int width, int type) throws IOException {
        long elements = (long) rows * width;
        if (type == F32) return Math.multiplyExact(elements, 4);
        if (type == Q8_0 && width % 32 == 0) return Math.multiplyExact(elements / 32, 34);
        throw new IOException("Unsupported GGUF tensor type or Q8_0 row width: " + type);
    }

    private static int alignment(Map<String, Object> fields) throws IOException {
        Object value = fields.getOrDefault("general.alignment", 32L);
        if (value instanceof Long n && n > 0 && n <= 4096 && (n & (n - 1)) == 0) return n.intValue();
        throw new IOException("Invalid GGUF data alignment");
    }

    private static long aligned(long offset, int alignment) {
        return Math.addExact(offset, alignment - 1) & ~(alignment - 1L);
    }

    private static Map<String, Entry> validated(Map<String, Entry> entries, long start, long size, int alignment)
            throws IOException {
        Map<String, Entry> result = new HashMap<>();
        var sorted = entries.entrySet().stream().sorted(Comparator.comparingLong(e -> e.getValue().offset())).toList();
        long end = start;
        for (var pair : sorted) {
            var tensor = pair.getValue().at(start);
            if (tensor.offset % alignment != 0 || tensor.offset < end || tensor.bytes > size - tensor.offset) {
                throw new IOException("Invalid GGUF range");
            }
            end = Math.addExact(tensor.offset, tensor.bytes);
            result.put(pair.getKey(), tensor);
        }
        return Map.copyOf(result);
    }
}
