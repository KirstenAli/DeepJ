package io.github.kirstenali.deepj.nn.transformer;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.persistence.ModelSettings;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.ModelResumeCase;
import java.io.*;
import java.nio.file.*;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static io.github.kirstenali.deepj.testing.NdModelFixtures.*;
import static org.junit.jupiter.api.Assertions.*;

class TypedModelLoaderTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void automaticLoadingPreservesAllFamiliesDtypesMetadataAndRawWeights(String family, DType dtype)
            throws IOException {
        DecoderOnlyModel original = model(family, 101, new ParameterFactory(dtype));
        original.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        original.setMetadata(Map.of("purpose", "nd-fixture", "seed", "101"));
        Path file = directory.resolve("model.dj");
        original.save(file);
        assertEquals(dtype == DType.FLOAT32 ? 3 : 5, ModelSerializer.formatVersion(file));
        compare(original, ModelLoader.load(file));
        assertEquals(original.parameters().stream().mapToLong(parameter -> parameter.value.numel()).sum(),
                ModelSerializer.inspect(file).parameters());
    }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#families")
    void mixedStorageResolvesSemanticNamesNotConstructorOrder(String family) throws IOException {
        DecoderOnlyModel source = model(family, 102);
        Map<String, TensorDescriptor> descriptors = new LinkedHashMap<>();
        for (var parameter : source.parameters()) {
            descriptors.put(parameter.name(), new TensorDescriptor(DType.values()[descriptors.size() % 3],
                    parameter.value.shape()));
        }
        DecoderOnlyModel original = model(family, 102, new ParameterFactory(descriptors));
        Path file = directory.resolve("mixed.dj");
        var reversed = new java.util.ArrayList<>(original.parameters());
        java.util.Collections.reverse(reversed);
        ModelSerializer.save(reversed, file, new ModelSettings(original.config(), original.promptFormat()));
        var layout = ModelSerializer.layout(file);
        assertEquals(descriptors, layout.parameters());
        assertThrows(UnsupportedOperationException.class, () -> layout.parameters().clear());
        compare(original, ModelLoader.load(file));
    }

    private static void compare(DecoderOnlyModel original, DecoderOnlyModel restored) throws IOException {
        assertEquals(original.getClass(), restored.getClass());
        assertEquals(original.config(), restored.config());
        assertEquals(original.promptFormat(), restored.promptFormat());
        assertEquals(original.metadata(), restored.metadata());
        assertEquals(original.parameters().size(), restored.parameters().size());
        for (int index = 0; index < original.parameters().size(); index++) {
            var first = original.parameters().get(index);
            var second = restored.parameters().get(index);
            assertEquals(first.name(), second.name());
            assertArrayEquals(encoded(first.value), encoded(second.value), "parameter=" + index);
        }
        try (var ignored = GradMode.noGrad()) {
            assertArrayEquals(original.forwardBatch(inputs()).toFloatArray(),
                    restored.forwardBatch(inputs()).toFloatArray());
        }
    }

    private static byte[] encoded(Tensor tensor) throws IOException {
        var bytes = new ByteArrayOutputStream();
        try (var output = new DataOutputStream(bytes)) { TensorEncoding.write(output, tensor); }
        return bytes.toByteArray();
    }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void actualModelFilesAndTrainingCheckpointsResumeLossesGradientsAndAllOptimizerStateExactly(
            String family, DType dtype) throws IOException {
        ModelResumeCase.verify(model(family, 103, new ParameterFactory(dtype)), directory);
    }
}
