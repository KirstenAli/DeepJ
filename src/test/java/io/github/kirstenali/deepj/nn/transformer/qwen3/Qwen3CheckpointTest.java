package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorCatalog;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorFile;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorWriter;
import io.github.kirstenali.deepj.tensor.DType;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3CheckpointTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest @EnumSource(DType.class)
    void configurationWeightsDtypesAndSubsequentGradientsSurviveRoundTrip(DType dtype) throws IOException {
        Qwen3 original = new Qwen3(Qwen3ModelTest.tiny(), 10, dtype);
        Path snapshot = directory.resolve("snapshot");
        Qwen3Checkpoint.save(original, snapshot);
        Qwen3 loaded = Qwen3Checkpoint.load(snapshot);
        assertEquals(original.config(), loaded.config());
        sameWeights(original, loaded);
        Tensor before = original.forward(new int[]{1, 2, 3}), after = loaded.forward(new int[]{1, 2, 3});
        assertArrayEquals(before.toFloatArray(), after.toFloatArray());
        after.sumTensor().backward();
        assertTrue(loaded.parameters().stream().allMatch(parameter -> parameter.grad() != null));
        assertThrows(java.nio.file.FileAlreadyExistsException.class, () -> Qwen3Checkpoint.save(loaded, snapshot));
    }

    private static void sameWeights(Qwen3 original, Qwen3 loaded) {
        for (int index = 0; index < original.parameters().size(); index++) {
            Tensor before = original.parameters().get(index).value, after = loaded.parameters().get(index).value;
            assertEquals(before.dtype(), after.dtype());
            assertArrayEquals(before.toFloatArray(), after.toFloatArray());
        }
    }

    @Test void shardedFilesAndAnIdenticalOptionalTiedHeadAreAccepted() throws IOException {
        Map<String, Tensor> tensors = tensors();
        tensors.put("lm_head.weight", tensors.get("model.embed_tokens.weight"));
        Map<String, Tensor> left = new LinkedHashMap<>(), right = new LinkedHashMap<>();
        Map<String, String> index = new LinkedHashMap<>();
        int position = 0;
        for (var entry : tensors.entrySet()) {
            boolean first = position++ % 2 == 0;
            (first ? left : right).put(entry.getKey(), entry.getValue());
            index.put(entry.getKey(), first ? "part-1.safetensors" : "part-2.safetensors");
        }
        SafeTensorWriter.write(directory.resolve("part-1.safetensors"), left);
        SafeTensorWriter.write(directory.resolve("part-2.safetensors"), right);
        Files.writeString(directory.resolve("model.safetensors.index.json"),
                Qwen3Json.stringify(Map.of("weight_map", index)));
        assertEquals(tensors.size(), SafeTensorCatalog.read(directory).size());
        assertNotNull(Qwen3Pretrained.load(directory));
    }

    @Test void missingUnknownAndUntiedWeightsAreRejectedBeforeModelConstruction() throws IOException {
        Map<String, Tensor> tensors = tensors();
        Tensor removed = tensors.remove("model.norm.weight");
        assertRejected(tensors);
        tensors.put("model.norm.weight", removed);
        tensors.put("unknown.weight", removed);
        assertRejected(tensors);
        tensors.remove("unknown.weight");
        tensors.put("lm_head.weight", tensors.get("model.embed_tokens.weight").addScalar(1));
        assertRejected(tensors);
    }

    @Test void indexCannotEscapeTheSnapshotDirectory() throws IOException {
        Files.writeString(directory.resolve("model.safetensors.index.json"),
                Qwen3Json.stringify(Map.of("weight_map", Map.of("a", "../outside.safetensors"))));
        assertThrows(IOException.class, () -> SafeTensorCatalog.read(directory));
    }

    private Map<String, Tensor> tensors() throws IOException {
        Qwen3Checkpoint.save(new Qwen3(Qwen3ModelTest.tiny(), 2), directory.resolve("source"));
        Files.copy(directory.resolve("source/config.json"), directory.resolve("config.json"));
        Map<String, Tensor> result = new LinkedHashMap<>();
        try (var ignored = GradMode.noGrad()) {
            for (var entry : SafeTensorCatalog.read(directory.resolve("source")).entrySet()) {
                result.put(entry.getKey(), SafeTensorFile.read(entry.getValue()));
            }
        }
        return result;
    }

    private void assertRejected(Map<String, Tensor> tensors) throws IOException {
        SafeTensorWriter.write(directory.resolve("model.safetensors"), tensors);
        assertThrows(IOException.class, () -> Qwen3Pretrained.load(directory));
    }
}
