package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.transformer.*;
import io.github.kirstenali.deepj.optimisers.ParameterFactory;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.testing.ModelResumeCase;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static io.github.kirstenali.deepj.testing.NdModelFixtures.*;
import static org.junit.jupiter.api.Assertions.*;

class MetalTypedModelLoaderTest extends MetalDifferentialTestSupport {
    @TempDir Path directory;

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void automaticDeviceLoadingPreservesPrecisionMetadataAndLogitBits(String family, DType dtype) throws IOException {
        DecoderOnlyModel original = model(family, 104, new ParameterFactory(dtype));
        roundTrip(original);
    }

    private void roundTrip(DecoderOnlyModel original) throws IOException {
        original.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        original.setMetadata(Map.of("source", "typed-gpu-fixture"));
        Path file = directory.resolve("model.dj");
        original.save(file);
        try (var ignored = GradMode.noGrad()) {
            float[] expected = original.forwardBatch(inputs()).toFloatArray();
            DecoderOnlyModel loaded = ModelLoader.load(file);
            assertArrayEquals(expected, loaded.forwardBatch(inputs()).toFloatArray());
            assertEquals(original.promptFormat(), loaded.promptFormat());
            assertEquals(original.metadata(), loaded.metadata());
            compareParameters(original, loaded);
        }
    }

    private static void compareParameters(DecoderOnlyModel original, DecoderOnlyModel loaded) {
        for (int index = 0; index < original.parameters().size(); index++) {
            Tensor first = original.parameters().get(index).value, second = loaded.parameters().get(index).value;
            assertEquals(first.dtype(), second.dtype());
            assertEquals(first.storageBytes(), second.storageBytes());
            assertArrayEquals(first.toFloatArray(), second.toFloatArray());
        }
    }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#families")
    void automaticDeviceLoadingRetainsMixedNamedStorageAndTiedParameters(String family) throws IOException {
        DecoderOnlyModel shapeReference = model(family, 105);
        Map<String, TensorDescriptor> descriptors = new LinkedHashMap<>();
        for (var parameter : shapeReference.parameters()) {
            descriptors.put(parameter.name(), new TensorDescriptor(DType.values()[descriptors.size() % 3],
                    parameter.value.shape()));
        }
        roundTrip(model(family, 105, new ParameterFactory(descriptors)));
        assertEquals(descriptors, ModelSerializer.layout(directory.resolve("model.dj")).parameters());
    }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void actualDeviceModelAndTrainingFilesResumeAllStateExactly(String family, DType dtype) throws IOException {
        ModelResumeCase.verify(model(family, 106, new ParameterFactory(dtype)), directory);
    }
}
