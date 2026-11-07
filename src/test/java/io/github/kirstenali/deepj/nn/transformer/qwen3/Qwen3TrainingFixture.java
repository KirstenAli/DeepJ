package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorWriter;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/** Shared initial weights and aligned targets for the optional training comparison. */
final class Qwen3TrainingFixture {
    private Qwen3TrainingFixture() {}

    static Qwen3 model() {
        return new Qwen3(new Qwen3Config(256, 128, 256, 4, 2, 2, 512, 128, 1_000_000, 0.02f, 1), 2026);
    }

    static int[] tokens(int seed) { return new Random(seed).ints(64, 0, 256).toArray(); }

    static void save(Qwen3 model, Path directory) throws IOException {
        Qwen3Checkpoint.save(model, directory);
        var input = Map.of("tokens", Arrays.stream(tokens(3)).boxed().toList(),
                "targets", Arrays.stream(tokens(4)).boxed().toList());
        Files.writeString(directory.resolve("input.json"), Qwen3Json.stringify(input));
        try (var ignored = GradMode.noGrad()) {
            SafeTensorWriter.write(directory.resolve("logits.safetensors"),
                    Map.of("logits", model.forward(tokens(3))));
        }
    }

    static Map<String, Tensor> gradients(Qwen3 model) {
        Map<String, Tensor> named = new LinkedHashMap<>(), result = new LinkedHashMap<>();
        model.parameters().forEach(parameter -> named.put(parameter.name(), parameter.grad()));
        for (var weight : Qwen3Weights.inventory(model.config())) {
            Tensor gradient = named.get(weight.name());
            result.put(weight.source(), weight.transpose()
                    ? gradient.transpose() : gradient.reshape(weight.sourceShape()));
        }
        return result;
    }
}
