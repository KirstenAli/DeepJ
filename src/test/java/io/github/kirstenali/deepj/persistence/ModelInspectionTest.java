package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ModelInspectionTest {
    @TempDir Path directory;

    @Test
    void inspectsShapesAndSettingsWithoutReadingOrChangingWeights() throws IOException {
        Path path = directory.resolve("model.dj");
        var parameters = List.of(new Parameter(Tensor.ones(2, 3)), new Parameter(Tensor.zeros(1, 4)));
        ModelSerializer.save(parameters, path, settings());
        byte[] before = Files.readAllBytes(path);
        ModelInspection result = ModelSerializer.inspect(path);
        assertEquals(settings(), result.settings());
        assertEquals(10, result.parameters());
        assertEquals(2, result.tensors());
        assertEquals(before.length, result.bytes());
        assertArrayEquals(before, Files.readAllBytes(path));
    }

    @Test
    void rejectsMissingSettingsAndEmptyTensorLists() throws IOException {
        Path path = directory.resolve("model.dj");
        ModelSerializer.save(List.of(new Parameter(Tensor.ones(1, 1))), path);
        assertThrows(IOException.class, () -> ModelSerializer.inspect(path));
        ModelSerializer.save(List.of(), path, settings());
        assertThrows(IOException.class, () -> ModelSerializer.inspect(path));
    }

    @Test
    void rejectsTruncatedPayloadAndTrailingData() throws IOException {
        Path path = directory.resolve("model.dj");
        ModelSerializer.save(List.of(new Parameter(Tensor.ones(2, 3))), path, settings());
        byte[] valid = Files.readAllBytes(path);
        Files.write(path, Arrays.copyOf(valid, valid.length - 1));
        assertThrows(IOException.class, () -> ModelSerializer.inspect(path));
        Files.write(path, Arrays.copyOf(valid, valid.length + 1));
        assertThrows(IOException.class, () -> ModelSerializer.inspect(path));
    }

    @Test
    void rejectsNegativeOrOverflowingTensorShapesBeforeSeeking() throws IOException {
        Path path = directory.resolve("model.dj");
        for (int[] shape : new int[][]{{0, 1}, {1, 0}, {-1, 4}, {Integer.MAX_VALUE, 2}}) {
            try (var output = new DataOutputStream(Files.newOutputStream(path))) {
                output.writeInt(0x444A4D44);
                output.writeInt(3);
                ModelSettingsIO.write(output, settings());
                output.writeInt(1);
                output.writeInt(shape[0]);
                output.writeInt(shape[1]);
            }
            assertThrows(IOException.class, () -> ModelSerializer.inspect(path));
        }
    }

    private ModelSettings settings() {
        return new ModelSettings(new DeepJPrismConfig(256, 32, 8, 2, 1, 16, 4, 2), PromptFormat.PLAIN);
    }
}
