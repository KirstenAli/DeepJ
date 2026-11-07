package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.persistence.qwen3.SafeTensorWriter;
import io.github.kirstenali.deepj.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** Untimed pre-clipping output for checking the independent Python workload. */
final class Qwen3TrainingAudit {
    private Qwen3TrainingAudit() {}

    static void save(Qwen3 model, Path output) throws IOException {
        Files.createDirectories(output);
        Tensor logits = model.forward(Qwen3TrainingFixture.tokens(3));
        SafeTensorWriter.write(output.resolve("logits.safetensors"), Map.of("logits", logits));
        Tensor loss = CrossEntropyLoss.forward(logits, Qwen3TrainingFixture.tokens(4), null);
        loss.backward();
        float norm = Tensor.backend().l2Norm(model.parameters().stream().map(parameter -> parameter.grad()).toList());
        SafeTensorWriter.write(output.resolve("gradients.safetensors"), Qwen3TrainingFixture.gradients(model));
        var result = Map.of("loss", loss.item(), "norm", norm);
        Files.writeString(output.resolve("result.json"), Qwen3Json.stringify(result));
    }
}
