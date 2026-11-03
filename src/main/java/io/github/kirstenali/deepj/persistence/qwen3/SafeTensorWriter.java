package io.github.kirstenali.deepj.persistence.qwen3;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorEncoding;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Logical-order safetensors export, retaining the exact FLOAT32/FLOAT16/BFLOAT16 storage bits. */
public final class SafeTensorWriter {
    private SafeTensorWriter() {}

    public static void write(Path path, Map<String, Tensor> tensors) throws IOException {
        var ordered = new LinkedHashMap<>(tensors);
        byte[] header = header(ordered);
        try (var output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            output.writeLong(Long.reverseBytes(header.length));
            output.write(header);
            for (Tensor tensor : ordered.values()) {
                values(output, tensor);
            }
        }
    }

    private static byte[] header(Map<String, Tensor> tensors) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        long offset = 0;
        for (var entry : tensors.entrySet()) {
            Tensor tensor = entry.getValue();
            long end = Math.addExact(offset, tensor.dtype().storageBytes(tensor.numel()));
            metadata.put(entry.getKey(), Map.of("dtype", dtype(tensor.dtype()),
                    "shape", java.util.Arrays.stream(tensor.shape()).boxed().toList(),
                    "data_offsets", List.of(offset, end)));
            offset = end;
        }
        String json = Qwen3Json.stringify(metadata);
        return (json + " ".repeat((8 - json.getBytes(StandardCharsets.UTF_8).length % 8) % 8))
                .getBytes(StandardCharsets.UTF_8);
    }

    private static String dtype(DType dtype) {
        return switch (dtype) {
            case FLOAT32 -> "F32";
            case FLOAT16 -> "F16";
            case BFLOAT16 -> "BF16";
        };
    }

    private static void values(DataOutputStream output, Tensor tensor) throws IOException {
        TensorEncoding.writeValues(new DataOutputStream(new EndianWords(output, tensor.dtype().bytes())), tensor);
    }
}
