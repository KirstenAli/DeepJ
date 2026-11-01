package io.github.kirstenali.deepj.persistence;

import io.github.kirstenali.deepj.nn.transformer.*;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class NamedModelFormatTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void namedFramesPreserveRawCompactPayloadsAndRequireExactNamesBeforeMutation() throws IOException {
        var parameters = parameters();
        Path file = directory.resolve("named.dj");
        ModelSerializer.save(parameters, file, settings());
        assertEquals(5, ModelSerializer.formatVersion(file));
        TensorDescriptor descriptor = new TensorDescriptor(DType.FLOAT16, new int[]{1, 2});
        assertEquals(descriptor, ModelSerializer.layout(file).parameters().get("first.weight"));
        assertThrows(IOException.class, () -> ModelSerializer.load(List.of(parameters.get(0),
                new Parameter("wrong", new Tensor(DType.BFLOAT16, 1, 2))), file));
        assertEquals(0, parameters.get(0).value.gradientState().version());
        var target = List.of(new Parameter("first.weight", new Tensor(DType.FLOAT16, 1, 2)),
                new Parameter("second.weight", new Tensor(DType.BFLOAT16, 1, 2)));
        ModelSerializer.load(target, file);
        assertEquals(0x80000000, Float.floatToRawIntBits(target.get(0).value.get(0, 0)));
        assertArrayEquals(parameters.get(1).value.toFloatArray(), target.get(1).value.toFloatArray());
    }

    @Test
    void namedDestinationsMustBeUniqueAndMatchAllRecordsBeforeAnyMutation() throws IOException {
        Path file = directory.resolve("names.dj");
        ModelSerializer.save(parameters(), file, settings());
        assertThrows(IOException.class, () -> ModelSerializer.load(List.of(
                new Parameter("", new Tensor(DType.FLOAT16, 1, 2)), parameters().get(1)), file));
        assertThrows(IOException.class, () -> ModelSerializer.load(List.of(
                parameters().get(0), new Parameter("first.weight", new Tensor(DType.BFLOAT16, 1, 2))), file));
        write(file, "first.weight", "first.weight", 2);
        var target = parameters();
        assertThrows(IOException.class, () -> ModelSerializer.load(target, file));
        assertEquals(0, target.get(0).value.gradientState().version());
        assertEquals(0, target.get(1).value.gradientState().version());
    }

    @Test
    void partialDuplicateAndInvalidNamesCannotReplaceAnExistingCheckpoint() throws IOException {
        Path file = directory.resolve("atomic.dj");
        ModelSerializer.save(parameters(), file, settings());
        byte[] before = Files.readAllBytes(file);
        for (String name : new String[]{"", "first.weight", "a..b", "wrong name", "x".repeat(1025)}) {
            var invalid = List.of(parameters().get(0), new Parameter(name, new Tensor(DType.BFLOAT16, 1, 2)));
            assertThrows(IOException.class, () -> ModelSerializer.save(invalid, file, settings()));
            assertArrayEquals(before, Files.readAllBytes(file));
        }
    }

    @Test
    void malformedNamedMetadataTruncationAndTrailingDataAreRejected() throws IOException {
        Path file = directory.resolve("corrupt.dj");
        for (String name : new String[]{"", "a..b", "x".repeat(1025)}) {
            write(file, name, "second.weight", 2);
            assertThrows(IOException.class, () -> ModelSerializer.layout(file));
        }
        write(file, "first.weight", "first.weight", 2);
        assertThrows(IOException.class, () -> ModelSerializer.inspect(file));
        ModelSerializer.save(parameters(), file, settings());
        byte[] bytes = Files.readAllBytes(file);
        for (int length : new int[]{bytes.length - 1, bytes.length + 1}) {
            Files.write(file, Arrays.copyOf(bytes, length));
            assertThrows(IOException.class, () -> ModelSerializer.layout(file));
        }
    }

    @Test
    void anonymousTypedFilesAreInspectableButNeverGuessedForAutomaticLoading() throws IOException {
        Path file = directory.resolve("anonymous.dj");
        ModelSerializer.save(List.of(new Parameter(new Tensor(DType.FLOAT16, 1, 2))), file, settings());
        assertEquals(4, ModelSerializer.formatVersion(file));
        assertEquals(2, ModelSerializer.inspect(file).parameters());
        assertThrows(IOException.class, () -> ModelLoader.load(file));
        write(file, "first.weight", "second.weight", 0);
        assertThrows(IOException.class, () -> ModelSerializer.layout(file));
        write(file, "first.weight", "second.weight", 3);
        assertThrows(IOException.class, () -> ModelSerializer.layout(file));
    }

    @Test
    void unknownSurplusOrWrongShapeDescriptorsAreNotSilentlyConverted() throws IOException {
        var original = io.github.kirstenali.deepj.testing.NdModelFixtures.model("origin", 42,
                new io.github.kirstenali.deepj.optimisers.ParameterFactory(DType.FLOAT16));
        Path file = directory.resolve("mismatch.dj");
        var parameters = new ArrayList<>(original.parameters());
        parameters.set(0, new Parameter("unknown.weight", parameters.get(0).value));
        ModelSerializer.save(parameters, file, new ModelSettings(original.config(), PromptFormat.PLAIN));
        assertThrows(IOException.class, () -> ModelLoader.load(file));
        parameters.set(0, original.parameters().get(0));
        parameters.add(new Parameter("surplus", new Tensor(DType.FLOAT16, 1, 2)));
        ModelSerializer.save(parameters, file, new ModelSettings(original.config(), PromptFormat.PLAIN));
        assertThrows(IOException.class, () -> ModelLoader.load(file));
        parameters.set(0, new Parameter("tokens.weight", new Tensor(DType.FLOAT16, 2, 2)));
        ModelSerializer.save(parameters, file, new ModelSettings(original.config(), PromptFormat.PLAIN));
        assertThrows(IOException.class, () -> ModelLoader.load(file));
    }

    private static void write(Path file, String first, String second, int count) throws IOException {
        try (var output = new DataOutputStream(Files.newOutputStream(file))) {
            ModelFormat.write(output, count, settings(), true, true);
            output.writeUTF(first);
            TensorEncoding.write(output, new Tensor(DType.FLOAT16, 1, 2));
            output.writeUTF(second);
            TensorEncoding.write(output, new Tensor(DType.BFLOAT16, 1, 2));
        }
    }

    private static List<Parameter> parameters() {
        return List.of(new Parameter("first.weight", Tensor.from(new float[]{-0.0f, 1.5f}, DType.FLOAT16, 1, 2)),
                new Parameter("second.weight", Tensor.from(new float[]{2.5f, -1.0f}, DType.BFLOAT16, 1, 2)));
    }

    private static ModelSettings settings() {
        return new ModelSettings(new DeepJPrismConfig(19, 8, 8, 2, 1, 16, 4, 3), PromptFormat.PLAIN);
    }
}
