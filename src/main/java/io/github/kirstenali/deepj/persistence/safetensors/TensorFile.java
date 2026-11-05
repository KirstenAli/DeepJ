package io.github.kirstenali.deepj.persistence.safetensors;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorFile;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.TensorDescriptor;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Adds convolution-rank metadata to the existing safetensors payload reader. */
final class TensorFile {
    private TensorFile() {}

    static Map<String, SafeTensorFile.Entry> inspect(Path path) throws IOException {
        try (var file = new RandomAccessFile(path.toFile(), "r")) {
            long length = Long.reverseBytes(file.readLong());
            if (length < 2 || length > 16 * 1024 * 1024 || length > file.length() - 8) {
                throw new IOException("Invalid safetensors header: " + path);
            }
            byte[] bytes = new byte[(int) length];
            file.readFully(bytes);
            var metadata = Qwen3Json.object(Qwen3Json.parse(new String(bytes, StandardCharsets.UTF_8)));
            return entries(path, metadata, length + 8, file.length());
        } catch (IllegalArgumentException | ArithmeticException failure) {
            throw new IOException("Invalid safetensors metadata: " + path, failure);
        }
    }

    private static Map<String, SafeTensorFile.Entry> entries(Path path, Map<String, Object> metadata,
                                                            long start, long size) throws IOException {
        Map<String, SafeTensorFile.Entry> result = new LinkedHashMap<>();
        metadata.forEach((name, value) -> {
            if (!name.equals("__metadata__")) result.put(name, entry(path, Qwen3Json.object(value), start));
        });
        long end = start;
        for (var entry : result.values().stream().sorted(Comparator.comparingLong(SafeTensorFile.Entry::offset))
                .toList()) {
            if (entry.offset() != end) throw new IOException("Safetensors payload has a gap or overlap");
            end = Math.addExact(end, entry.descriptor().bytes());
        }
        if (result.isEmpty() || end != size) throw new IOException("Safetensors payload length mismatch");
        return Map.copyOf(result);
    }

    private static SafeTensorFile.Entry entry(Path path, Map<String, Object> value, long start) {
        int[] shape = Qwen3Json.array(value.get("shape")).stream()
                .mapToInt(v -> Math.toIntExact(Qwen3Json.integer(v))).toArray();
        if (shape.length < 1 || shape.length > 4) throw new IllegalArgumentException("Invalid weight rank");
        for (int size : shape) {
            if (size < 1) throw new IllegalArgumentException("Invalid weight dimension");
        }
        var descriptor = new TensorDescriptor(dtype(Qwen3Json.string(value.get("dtype"))), shape);
        List<?> offsets = Qwen3Json.array(value.get("data_offsets"));
        if (offsets.size() != 2) throw new IllegalArgumentException("Invalid weight range");
        long begin = Qwen3Json.integer(offsets.get(0)), end = Qwen3Json.integer(offsets.get(1));
        if (begin < 0 || Math.subtractExact(end, begin) != descriptor.bytes()) {
            throw new IllegalArgumentException("Weight shape and byte count disagree");
        }
        return new SafeTensorFile.Entry(path, descriptor, Math.addExact(start, begin));
    }

    private static DType dtype(String value) {
        return switch (value) {
            case "F32" -> DType.FLOAT32;
            case "F16" -> DType.FLOAT16;
            case "BF16" -> DType.BFLOAT16;
            default -> throw new IllegalArgumentException("Unsupported weight dtype: " + value);
        };
    }
}
