package io.github.kirstenali.deepj.persistence.safetensors;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class TensorFileVideoTest {
    @TempDir Path directory;

    @Test void fiveDimensionalWeightsLoadWithoutChangingPayloadOrDimensions() throws Exception {
        Tensor.setBackend(new CpuBackend());
        write("[1,1,2,1,2]", 16);
        try (var weights = TensorWeights.open(directory, DType.FLOAT32); var ignored = GradMode.noGrad()) {
            Tensor value = weights.tensor("weight");
            assertArrayEquals(new int[]{1, 1, 2, 1, 2}, value.shape());
            assertArrayEquals(new float[]{1, 2, 3, 4}, value.toFloatArray());
        }
    }

    @Test void higherRanksAndMismatchedLengthsRemainInvalid() throws Exception {
        write("[1,1,1,2,1,2]", 16);
        assertThrows(IOException.class, () -> TensorWeights.open(directory, DType.FLOAT32));
        write("[1,1,2,1,2]", 12);
        assertThrows(IOException.class, () -> TensorWeights.open(directory, DType.FLOAT32));
    }

    private void write(String shape, int length) throws IOException {
        String json = "{\"weight\":{\"dtype\":\"F32\",\"shape\":" + shape
                + ",\"data_offsets\":[0," + length + "]}}";
        byte[] header = json.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buffer = ByteBuffer.allocate(8 + header.length + 16).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(header.length).put(header);
        for (int i = 1; i <= 4; i++) { buffer.putFloat(i); }
        Files.write(directory.resolve("model.safetensors"), buffer.array());
    }
}
