package io.github.kirstenali.deepj.persistence.qwen3;

import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class SafeTensorFileTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void nonContiguousViewsRetainLogicalOrderAndCompactBits(DType dtype) throws IOException {
        Tensor tensor = Tensor.from(new float[]{-1.25f, 0.003f, 23, -0.0f, 1.0078125f, 5}, 2, 3).to(dtype);
        Tensor view = tensor.transpose().slice(0, 0, 2);
        Path path = directory.resolve("weights");
        SafeTensorWriter.write(path, Map.of("view", view));
        Tensor read = SafeTensorFile.read(SafeTensorFile.inspect(path).get("view"));
        assertEquals(dtype, read.dtype());
        assertArrayEquals(view.shape(), read.shape());
        assertArrayEquals(view.toFloatArray(), read.toFloatArray());
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "{\"x\":{\"dtype\":\"I8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[2147483647,2],\"data_offsets\":[0,4]}}",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}}",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[1,5]}}",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,8]}}",
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
                    + "\"y\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}"})
    void invalidMetadataCannotAllocateOrSilentlyOverlapWeights(String header) throws IOException {
        Path path = write(header, 4);
        assertThrows(IOException.class, () -> SafeTensorFile.inspect(path));
    }

    @Test void truncatedPayloadAndOversizedHeaderAreRejected() throws IOException {
        Path path = write("{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", 3);
        assertThrows(IOException.class, () -> SafeTensorFile.inspect(path));
        Files.write(path, ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(Long.MAX_VALUE).array());
        assertThrows(IOException.class, () -> SafeTensorFile.inspect(path));
    }

    private Path write(String header, int length) throws IOException {
        byte[] bytes = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer.allocate(8 + bytes.length + length).order(ByteOrder.LITTLE_ENDIAN);
        Path path = directory.resolve("invalid");
        Files.write(path, buffer.putLong(bytes.length).put(bytes).array());
        return path;
    }
}
