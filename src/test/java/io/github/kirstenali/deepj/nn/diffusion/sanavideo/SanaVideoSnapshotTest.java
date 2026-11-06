package io.github.kirstenali.deepj.nn.diffusion.sanavideo;

import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class SanaVideoSnapshotTest {
    private static final Path MODEL = Path.of("src/test/resources/sana-video/pipeline");
    @TempDir Path directory;

    @Test void officialDiffusersShardIndexOpensThroughTheSharedReader() throws Exception {
        try (var paths = Files.walk(MODEL)) {
            for (Path path : paths.toList()) {
                Path target = directory.resolve(MODEL.relativize(path));
                if (Files.isDirectory(path)) Files.createDirectories(target);
                else Files.copy(path, target);
            }
        }
        Path stage = directory.resolve("transformer");
        String index;
        try (var weights = TensorWeights.open(stage, DType.FLOAT32)) {
            index = weights.descriptors().keySet().stream().sorted()
                    .map(name -> "\"" + name + "\":\"model.safetensors\"").collect(Collectors.joining(","));
        }
        Files.writeString(stage.resolve("diffusion_pytorch_model.safetensors.index.json"),
                "{\"weight_map\":{" + index + "}}");
        SanaVideoSnapshot.validate(directory);
        assertEquals(Files.readString(stage.resolve("diffusion_pytorch_model.safetensors.index.json")),
                Files.readString(stage.resolve("model.safetensors.index.json")));
    }
}
