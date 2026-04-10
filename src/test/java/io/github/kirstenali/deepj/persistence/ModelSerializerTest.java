package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorAdapters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ModelSerializerTest {

    @TempDir
    Path tempDir;

    @Test
    void saveAndLoad_roundTripsParameterValues() throws IOException {
        List<Parameter> original = List.of(
                new Parameter(Tensor.from2D(new float[][]{
                        {1.0f, 2.0f},
                        {3.0f, 4.0f}
                })),
                new Parameter(rowTensor(5.0f, 6.0f, 7.0f))
        );

        Path file = tempDir.resolve("model.bin");
        ModelSerializer.save(original, file);

        List<Parameter> loaded = List.of(
                new Parameter(Tensor.zeros(2, 2)),
                new Parameter(Tensor.zeros(1, 3))
        );

        ModelSerializer.load(loaded, file);

        assertTensorEquals(original.get(0).value, loaded.get(0).value, 1e-12f);
        assertTensorEquals(original.get(1).value, loaded.get(1).value, 1e-12f);
    }

    @Test
    void save_createsParentDirectories() throws IOException {
        List<Parameter> params = List.of(
                new Parameter(rowTensor(1.0f))
        );

        Path file = tempDir.resolve("nested").resolve("dir").resolve("model.bin");
        ModelSerializer.save(params, file);

        assertTrue(java.nio.file.Files.exists(file));
    }

    @Test
    void load_throwsWhenParameterCountMismatch() throws IOException {
        List<Parameter> saved = List.of(
                new Parameter(rowTensor(1.0f)),
                new Parameter(rowTensor(2.0f))
        );

        Path file = tempDir.resolve("model.bin");
        ModelSerializer.save(saved, file);

        List<Parameter> target = List.of(
                new Parameter(Tensor.zeros(1, 1))
        );

        IOException ex = assertThrows(IOException.class, () -> ModelSerializer.load(target, file));
        assertTrue(ex.getMessage().contains("Parameter count mismatch"));
    }

    @Test
    void load_throwsWhenShapeMismatch() throws IOException {
        List<Parameter> saved = List.of(
                new Parameter(Tensor.from2D(new float[][]{
                        {1.0f, 2.0f},
                        {3.0f, 4.0f}
                }))
        );

        Path file = tempDir.resolve("model.bin");
        ModelSerializer.save(saved, file);

        List<Parameter> target = List.of(
                new Parameter(Tensor.zeros(1, 3))
        );

        IOException ex = assertThrows(IOException.class, () -> ModelSerializer.load(target, file));
        assertTrue(ex.getMessage().contains("Shape mismatch"));
    }

    private static void assertTensorEquals(Tensor expected, Tensor actual, double tol) {
        assertEquals(expected.rows, actual.rows, "rows");
        assertEquals(expected.cols, actual.cols, "cols");

        for (int r = 0; r < expected.rows; r++) {
            for (int c = 0; c < expected.cols; c++) {
                assertEquals(expected.data[r * expected.cols + c], actual.data[r * actual.cols + c], tol,
                        "mismatch at [" + r + "," + c + "]");
            }
        }
    }

    private static Tensor rowTensor(float... values) {
        return TensorAdapters.unpackF32(values, 1, values.length);
    }
}