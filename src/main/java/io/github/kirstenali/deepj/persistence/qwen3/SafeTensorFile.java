package io.github.kirstenali.deepj.persistence.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Validates floating-point safetensors metadata before allocating any tensor payloads. */
public final class SafeTensorFile {
    private SafeTensorFile() {}

    public record Entry(Path file, TensorDescriptor descriptor, long offset) {}

    public static Map<String, Entry> inspect(Path path) throws IOException {
        try (var file = new RandomAccessFile(path.toFile(), "r")) {
            long length = Long.reverseBytes(file.readLong());
            if (length < 2 || length > 16 * 1024 * 1024 || length > file.length() - 8) {
                throw new IOException("Invalid safetensors header length");
            }
            byte[] bytes = new byte[(int) length];
            file.readFully(bytes);
            String json = StandardCharsets.UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString();
            Map<String, Entry> entries = entries(path, Qwen3Json.object(Qwen3Json.parse(json)), length + 8);
            requireCoverage(entries.values().stream().toList(), file.length(), length + 8);
            return Map.copyOf(entries);
        } catch (IllegalArgumentException | ArithmeticException failure) {
            throw new IOException("Invalid safetensors metadata: " + path, failure);
        }
    }

    private static Map<String, Entry> entries(Path path, Map<String, Object> header, long start) {
        Map<String, Entry> result = new LinkedHashMap<>();
        header.forEach((name, value) -> {
            if (!name.equals("__metadata__")) result.put(name, entry(path, Qwen3Json.object(value), start));
        });
        if (result.isEmpty()) throw new IllegalArgumentException("No weights in safetensors file");
        return result;
    }

    private static Entry entry(Path path, Map<String, Object> metadata, long start) {
        DType dtype = dtype(Qwen3Json.string(metadata.get("dtype")));
        int[] shape = Qwen3Json.array(metadata.get("shape")).stream()
                .mapToInt(value -> Math.toIntExact(Qwen3Json.integer(value))).toArray();
        if (shape.length < 1 || shape.length > 2) throw new IllegalArgumentException("Expected vector or matrix");
        for (int dimension : shape) {
            if (dimension < 1) throw new IllegalArgumentException("Empty weight dimension");
        }
        TensorDescriptor descriptor = new TensorDescriptor(dtype, shape);
        List<?> offsets = Qwen3Json.array(metadata.get("data_offsets"));
        if (offsets.size() != 2) throw new IllegalArgumentException("Expected two data offsets");
        long begin = Qwen3Json.integer(offsets.get(0)), end = Qwen3Json.integer(offsets.get(1));
        if (begin < 0 || Math.subtractExact(end, begin) != descriptor.bytes()) {
            throw new IllegalArgumentException("Weight byte range does not match its shape");
        }
        return new Entry(path, descriptor, Math.addExact(start, begin));
    }

    private static void requireCoverage(List<Entry> entries, long fileLength, long start) throws IOException {
        long expected = start;
        for (Entry entry : entries.stream().sorted(Comparator.comparingLong(Entry::offset)).toList()) {
            if (entry.offset() != expected) throw new IOException("Safetensors data contains gaps or overlaps");
            expected = Math.addExact(expected, entry.descriptor().bytes());
        }
        if (expected != fileLength) throw new IOException("Safetensors payload length mismatch");
    }

    private static DType dtype(String name) {
        return switch (name) {
            case "F32" -> DType.FLOAT32;
            case "F16" -> DType.FLOAT16;
            case "BF16" -> DType.BFLOAT16;
            default -> throw new IllegalArgumentException("Unsupported weight dtype: " + name);
        };
    }

    public static Tensor read(Entry entry) throws IOException {
        Tensor result = new Tensor(entry.descriptor().dtype(), entry.descriptor().shape());
        try (var file = new RandomAccessFile(entry.file().toFile(), "r")) {
            file.seek(entry.offset());
            TensorEncoding.readValues(file, result, ByteOrder.LITTLE_ENDIAN);
        }
        return result;
    }

    /** Restores matching CPU storage directly; device transfers and dtype conversions retain private staging. */
    public static void restore(Entry entry, Tensor target, boolean transpose) throws IOException {
        if (Tensor.backend().usesDeviceStorage() || target.dtype() != entry.descriptor().dtype()) {
            TensorEncoding.restoreWeight(target, read(entry), transpose);
            return;
        }
        try (var ignored = GradMode.noGrad(); var file = new RandomAccessFile(entry.file().toFile(), "r")) {
            Tensor view = transpose ? target.transpose(0, 1) : target.view(entry.descriptor().shape());
            entry.descriptor().requireMatches(view);
            file.seek(entry.offset());
            TensorEncoding.readValues(file, view, ByteOrder.LITTLE_ENDIAN);
        }
    }

    public static void requireIdentical(Entry left, Entry right) throws IOException {
        if (!left.descriptor().equals(right.descriptor())) throw new IOException("Tied embedding metadata differs");
        try (var first = new RandomAccessFile(left.file().toFile(), "r");
             var second = new RandomAccessFile(right.file().toFile(), "r")) {
            first.seek(left.offset());
            second.seek(right.offset());
            compareChunks(first, second, left.descriptor().bytes());
        }
    }

    private static void compareChunks(RandomAccessFile first, RandomAccessFile second, long remaining)
            throws IOException {
        byte[] left = new byte[1 << 18], right = new byte[left.length];
        while (remaining > 0) {
            int length = (int) Math.min(left.length, remaining);
            first.readFully(left, 0, length);
            second.readFully(right, 0, length);
            if (!java.util.Arrays.equals(left, 0, length, right, 0, length)) {
                throw new IOException("lm_head differs from the tied token embedding");
            }
            remaining -= length;
        }
    }
}
