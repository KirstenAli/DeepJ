package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
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
                new Parameter(Tensor.from2D(new double[][]{
                        {1.0, 2.0},
                        {3.0, 4.0}
                })),
                new Parameter(Tensor.from2D(new double[][]{
                        {5.0, 6.0, 7.0}
                }))
        );

        Path file = tempDir.resolve("model.bin");
        ModelSerializer.save(original, file);

        List<Parameter> loaded = List.of(
                new Parameter(Tensor.from2D(new double[][]{
                        {0.0, 0.0},
                        {0.0, 0.0}
                })),
                new Parameter(Tensor.from2D(new double[][]{
                        {0.0, 0.0, 0.0}
                }))
        );

        ModelSerializer.load(loaded, file);

        assertTensorEquals(original.get(0).value, loaded.get(0).value, 1e-12);
        assertTensorEquals(original.get(1).value, loaded.get(1).value, 1e-12);
    }

    @Test
    void save_createsParentDirectories() throws IOException {
        List<Parameter> params = List.of(
                new Parameter(Tensor.from2D(new double[][]{{1.0}}))
        );

        Path file = tempDir.resolve("nested").resolve("dir").resolve("model.bin");
        ModelSerializer.save(params, file);

        assertTrue(java.nio.file.Files.exists(file));
    }

    @Test
    void load_throwsWhenParameterCountMismatch() throws IOException {
        List<Parameter> saved = List.of(
                new Parameter(Tensor.from2D(new double[][]{{1.0}})),
                new Parameter(Tensor.from2D(new double[][]{{2.0}}))
        );

        Path file = tempDir.resolve("model.bin");
        ModelSerializer.save(saved, file);

        List<Parameter> target = List.of(
                new Parameter(Tensor.from2D(new double[][]{{0.0}}))
        );

        IOException ex = assertThrows(IOException.class, () -> ModelSerializer.load(target, file));
        assertTrue(ex.getMessage().contains("Parameter count mismatch"));
    }

    @Test
    void load_throwsWhenShapeMismatch() throws IOException {
        List<Parameter> saved = List.of(
                new Parameter(Tensor.from2D(new double[][]{
                        {1.0, 2.0},
                        {3.0, 4.0}
                }))
        );

        Path file = tempDir.resolve("model.bin");
        ModelSerializer.save(saved, file);

        List<Parameter> target = List.of(
                new Parameter(Tensor.from2D(new double[][]{
                        {0.0, 0.0, 0.0}
                }))
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
}