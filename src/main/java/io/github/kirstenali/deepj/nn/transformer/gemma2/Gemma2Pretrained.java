package io.github.kirstenali.deepj.nn.transformer.gemma2;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.persistence.ModelInspection;
import io.github.kirstenali.deepj.persistence.ModelSettings;
import io.github.kirstenali.deepj.persistence.safetensors.TensorWeights;
import io.github.kirstenali.deepj.tensor.DType;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/** Imports standard safetensors checkpoints, including the shared SANA Gemma backbone weights. */
public final class Gemma2Pretrained {
    private Gemma2Pretrained() {}

    public static Gemma2 load(Path directory, DType dtype) throws IOException {
        var config = Gemma2Config.read(directory.resolve("config.json"));
        var weights = TensorWeights.open(directory, dtype == null ? DType.BFLOAT16 : dtype);
        try {
            Gemma2Layout.validate(config, weights.descriptors());
            return new Gemma2(config, weights);
        } catch (IOException | RuntimeException | Error failure) {
            weights.close();
            throw failure;
        }
    }

    public static ModelInspection inspect(Path directory) throws IOException {
        var config = Gemma2Config.read(directory.resolve("config.json"));
        try (var weights = TensorWeights.open(directory, DType.BFLOAT16)) {
            var descriptors = weights.descriptors();
            Gemma2Layout.validate(config, descriptors);
            long parameters = 0, bytes = 0;
            for (var descriptor : descriptors.values()) {
                bytes = Math.addExact(bytes, descriptor.bytes());
                parameters = Math.addExact(parameters, descriptor.bytes() / descriptor.dtype().bytes());
            }
            return new ModelInspection(new ModelSettings(config, PromptFormat.PLAIN,
                    Map.of("name", "Gemma 2", "prompt_format", "Gemma 2 chat")), parameters, descriptors.size(), bytes);
        }
    }
}
