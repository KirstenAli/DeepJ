package io.github.kirstenali.deepj.onnx;

import static org.junit.jupiter.api.Assertions.*;

import io.github.kirstenali.deepj.autograd.GradMode;
import io.github.kirstenali.deepj.nn.NetworkWeights;
import io.github.kirstenali.deepj.optimisers.AdamW;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorScope;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

class OptimizerPersistenceTest {
    @TempDir Path directory;

    @BeforeAll
    static void backend() {
        ReferenceFixturesTest.backend();
    }

    @AfterAll
    static void release() {
        ReferenceFixturesTest.release();
    }

    @Test
    void existingOptimizerAndWeightArchivesWorkWithImportedParameters() throws Exception {
        Path path = ReferenceFixturesTest.FIXTURES.resolve("mlp.onnx");
        try (var scope = TensorScope.open();
                var model = OnnxModel.load(path)) {
            Tensor input = input();
            assertTrue(model.parameters().isEmpty(), "Inference must not allocate trainable parameters");
            model.trainable("W", "B");
            float before = loss(model, input).item();
            loss(model, input).backward();
            AdamW.defaultAdamW(.001f).step(model.parameters());
            model.zeroGrad();
            assertTrue(loss(model, input).item() < before);
            assertTrue(model.parameters().stream().allMatch(p -> p.grad().sumAbs() == 0));
            roundTrip(path, model, input);
        }
    }

    private void roundTrip(Path path, OnnxModel model, Tensor input) throws Exception {
        Path weights = directory.resolve("network.djnn");
        NetworkWeights.save(weights, model);
        try (var restored = OnnxModel.load(path);
                var ignored = GradMode.noGrad()) {
            restored.trainable("W", "B");
            NetworkWeights.load(weights, restored);
            assertArrayEquals(model.forward(input).toFloatArray(), restored.forward(input).toFloatArray());
        }
    }

    private static Tensor loss(OnnxModel model, Tensor input) {
        Tensor output = model.forward(input);
        return output.multiply(output).sumTensor();
    }

    private static Tensor input() throws Exception {
        var fixture = Qwen3Json.read(ReferenceFixturesTest.FIXTURES.resolve("mlp.json"));
        return ReferenceFixturesTest.input(Qwen3Json.object(fixture.get("inputs")).get("X")).tensor();
    }
}
