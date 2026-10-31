package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorAdapters;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
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

    @Test
    void rejectsHeaderlessDoubleCheckpointsWithoutChangingParameters() throws IOException {
        Path file = tempDir.resolve("unsupported-double.bin");
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
            out.writeInt(1);
            out.writeInt(1);
            out.writeInt(2);
            out.writeDouble(1.25);
            out.writeDouble(-2.5);
        }

        Parameter parameter = new Parameter(rowTensor(7, 8));
        IOException error = assertThrows(IOException.class, () -> ModelSerializer.load(List.of(parameter), file));
        assertEquals("Unsupported checkpoint format: missing DeepJ header", error.getMessage());
        assertArrayEquals(new float[]{7, 8}, parameter.value.rawData());
        assertThrows(IOException.class, () -> ModelSerializer.settings(file));
        assertThrows(IOException.class, () -> ModelSerializer.inspect(file));
    }

    @Test
    void newFormatUsesFloat32Payloads() throws IOException {
        Path file = tempDir.resolve("compact.bin");
        ModelSerializer.save(List.of(new Parameter(rowTensor(1.0f, 2.0f))), file);

        assertEquals(28L, Files.size(file));
    }

    @Test
    void rejectsVersionTwoCheckpoints() throws IOException {
        Path file = tempDir.resolve("version-two.dj");
        try (DataOutputStream out = new DataOutputStream(Files.newOutputStream(file))) {
            out.writeInt(0x444A4D44);
            out.writeInt(2);
        }
        IOException error = assertThrows(IOException.class, () -> ModelSerializer.settings(file));
        assertEquals("Unsupported model format version: 2", error.getMessage());
    }

    private static void assertTensorEquals(Tensor expected, Tensor actual, double tol) {
        assertEquals(expected.matrixRows(), actual.matrixRows(), "rows");
        assertEquals(expected.matrixCols(), actual.matrixCols(), "cols");

        for (int r = 0; r < expected.matrixRows(); r++) {
            for (int c = 0; c < expected.matrixCols(); c++) {
                assertEquals(expected.rawData()[r * expected.matrixCols() + c],
                        actual.rawData()[r * actual.matrixCols() + c], tol,
                        "mismatch at [" + r + "," + c + "]");
            }
        }
    }

    private static Tensor rowTensor(float... values) {
        return TensorAdapters.unpackF32(values, 1, values.length);
    }
}
