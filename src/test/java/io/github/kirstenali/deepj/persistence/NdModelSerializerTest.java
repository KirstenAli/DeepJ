package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.*;
import java.nio.file.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

class NdModelSerializerTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void mixedRankAndDtypeFilesPreserveStorageWithoutImplicitConversion() throws IOException {
        var source = parameters();
        var target = source.stream().map(parameter -> new Parameter(
                new Tensor(parameter.value.dtype(), parameter.value.shape()))).toList();
        Path file = directory.resolve("typed.dj");
        ModelSerializer.save(source, file);
        assertVersion(file, 4);
        ModelSerializer.load(target, file);
        for (int index = 0; index < source.size(); index++) {
            assertArrayEquals(payload(source.get(index).value), payload(target.get(index).value));
        }
        assertNull(ModelSerializer.settings(file));
    }

    @Test
    void embeddedSettingsAndInspectionUseTypedPhysicalLengths() throws IOException {
        var parameters = parameters();
        Path file = directory.resolve("embedded.dj");
        ModelSerializer.save(parameters, file, settings());
        assertEquals(settings(), ModelSerializer.settings(file));
        ModelInspection inspection = ModelSerializer.inspect(file);
        assertEquals(parameters.stream().mapToLong(parameter -> parameter.value.numel()).sum(),
                inspection.parameters());
        assertEquals(parameters.size(), inspection.tensors());
        assertEquals(Files.size(file), inspection.bytes());
        assertEquals(settings(), inspection.settings());
        ModelSerializer.load(parameters, file, settings().config());
    }

    @Test
    void truncationTrailingBytesAndDtypeMismatchesCannotPartiallyChangeWeights() throws IOException {
        Path file = directory.resolve("invalid.dj");
        var source = List.of(new Parameter(Tensor.from(new float[]{1, 2}, DType.FLOAT16)),
                new Parameter(Tensor.from(new float[]{3, 4}, DType.BFLOAT16)));
        ModelSerializer.save(source, file);
        byte[] valid = Files.readAllBytes(file);
        var target = List.of(new Parameter(Tensor.from(new float[]{7, 8}, DType.FLOAT16)),
                new Parameter(Tensor.from(new float[]{9, 10}, DType.BFLOAT16)));
        Files.write(file, Arrays.copyOf(valid, valid.length - 1));
        assertRejectedWithoutMutation(target, file);
        Files.write(file, Arrays.copyOf(valid, valid.length + 1));
        assertRejectedWithoutMutation(target, file);
        Files.write(file, valid);
        assertThrows(IOException.class, () -> ModelSerializer.load(List.of(
                new Parameter(new Tensor(2)), target.get(1)), file));
        assertArrayEquals(new float[]{9, 10}, target.get(1).value.toFloatArray());
    }

    private static void assertRejectedWithoutMutation(List<Parameter> target, Path file) {
        assertThrows(IOException.class, () -> ModelSerializer.load(target, file));
        assertArrayEquals(new float[]{7, 8}, target.get(0).value.toFloatArray());
        assertArrayEquals(new float[]{9, 10}, target.get(1).value.toFloatArray());
        assertEquals(0, target.get(0).value.gradientState().version());
        assertEquals(0, target.get(1).value.gradientState().version());
    }

    @Test
    void originalMatrixFramingStillWritesTheSameFinitePayloadAndPreservesNanBits() throws IOException {
        Tensor matrix = Tensor.from(new float[]{-0.0f, Float.intBitsToFloat(0x7f812345)}, 1, 2);
        Path file = directory.resolve("matrix.dj");
        ModelSerializer.save(List.of(new Parameter(matrix)), file);
        assertVersion(file, 1);
        assertEquals(28, Files.size(file));
        var target = new Parameter(new Tensor(1, 2));
        ModelSerializer.load(List.of(target), file);
        assertEquals(0x80000000, Float.floatToRawIntBits(target.value.get(0, 0)));
        assertEquals(0x7f812345, Float.floatToRawIntBits(target.value.get(0, 1)));
        ModelSerializer.save(List.of(target), file, settings());
        assertVersion(file, 3);
    }

    @Test
    void typedHeaderCorruptionIsRejectedByInspectionBeforeAllocatingWeights() throws IOException {
        Path file = directory.resolve("corrupt.dj");
        ModelSerializer.save(List.of(new Parameter(new Tensor(DType.FLOAT16, 2))), file, settings());
        byte[] valid = Files.readAllBytes(file);
        Files.write(file, Arrays.copyOf(valid, valid.length - 1));
        assertThrows(IOException.class, () -> ModelSerializer.inspect(file));
        byte[] trailing = Arrays.copyOf(valid, valid.length + 1);
        Files.write(file, trailing);
        assertThrows(IOException.class, () -> ModelSerializer.inspect(file));
    }

    private static List<Parameter> parameters() {
        return List.of(new Parameter(Tensor.from(new float[]{0.7f}, new int[0])),
                new Parameter(Tensor.from(new float[]{1.3f, -0.2f}, DType.FLOAT16)),
                new Parameter(Tensor.from(new float[]{1, 2, 3, 4}, DType.BFLOAT16, 2, 1, 2)),
                new Parameter(new Tensor(0, 3)), new Parameter(Tensor.from(new float[]{1, 2, 3, 4, 5, 6}, 3, 2)
                .slice(0, 1, 3).transpose(0, 1)));
    }

    private static byte[] payload(Tensor tensor) throws IOException {
        var bytes = new ByteArrayOutputStream();
        try (var output = new DataOutputStream(bytes)) { TensorEncoding.write(output, tensor); }
        return bytes.toByteArray();
    }

    private static void assertVersion(Path path, int version) throws IOException {
        try (var input = new DataInputStream(Files.newInputStream(path))) {
            assertEquals(0x444A4D44, input.readInt());
            assertEquals(version, input.readInt());
        }
    }

    private static ModelSettings settings() {
        return new ModelSettings(new DeepJPrismConfig(256, 32, 8, 2, 1, 16, 4, 2),
                PromptFormat.INSTRUCTION_RESPONSE, Map.of("fixture", "nd-checkpoint"));
    }
}
