package io.github.kirstenali.deepj.models.nova;

import io.github.kirstenali.deepj.models.ModelLoader;
import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DeepJNovaTest {

    @TempDir Path directory;

    @BeforeEach
    void useCpu() {
        Tensor.setBackend(new CpuBackend());
    }

    @Test
    void parameterCountMatchesUniqueTrainableWeights() {
        var config = new DeepJNovaConfig(11, 6, 8, 4, 2, 2, 12);
        var model = new DeepJNova(config, 42L);
        long actual = model.parameters().stream().mapToLong(p -> p.value.data.length).sum();
        assertEquals(DeepJNovaParameterCount.count(config), actual);
        assertEquals(model.parameters().size(), model.parameters().stream().distinct().count());
        assertEquals(1, model.parameters().stream().filter(p -> p.value.rows == 11).count());
    }

    @Test
    void candidateHasApproximatelyOneHundredMillionParameters() {
        var config = new DeepJNovaConfig(16_384, 1_024, 576, 9, 3, 24, 1_664);
        assertEquals(99_711_552L, DeepJNovaParameterCount.count(config));
    }

    @Test
    void rejectsInvalidKeyValueGrouping() {
        assertThrows(IllegalArgumentException.class, () -> new DeepJNovaConfig(11, 6, 8, 4, 0, 2, 12));
        assertThrows(IllegalArgumentException.class, () -> new DeepJNovaConfig(11, 6, 8, 4, 3, 2, 12));
        assertThrows(IllegalArgumentException.class, () -> new DeepJNovaConfig(11, 6, 8, 4, 8, 2, 12));
    }

    @Test
    void futureTokensCannotChangeEarlierLogits() {
        var model = new DeepJNova(new DeepJNovaConfig(11, 6, 8, 4, 2, 2, 12), 42L);
        float[] before = model.forward(new int[]{1, 2, 3}).data.clone();
        float[] after = model.forward(new int[]{1, 2, 7}).data;
        assertArrayEquals(Arrays.copyOf(before, 22), Arrays.copyOf(after, 22), 1e-6f);
    }

    @Test
    void optimizerUpdatesLearnARepeatedBatchWithTiedWeights() {
        var model = new DeepJNova(new DeepJNovaConfig(11, 4, 8, 4, 2, 2, 12), 42L);
        int[] input = {1, 3, 5, 2}, target = {3, 5, 2, 1};
        Batch batch = new Batch(new int[][]{input}, new int[][]{target});
        float before = CrossEntropyLoss.loss(model.forward(input), target);
        var trainer = CausalLMTraining.trainer(model, ignored -> batch, 1e-2f);
        for (int step = 0; step < 50; step++) {
            assertTrue(Float.isFinite(trainer.trainStep(1)));
        }
        assertTrue(CrossEntropyLoss.loss(model.forward(input), target) < before * 0.6f);
    }

    @Test
    void checkpointRestoresArchitecturePromptAndOutputs() throws Exception {
        var config = new DeepJNovaConfig(11, 6, 8, 4, 2, 2, 12, false, 0.3f, 0.7f);
        var model = new DeepJNova(config, 42L);
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        model.setMetadata(Map.of("purpose", "round-trip"));
        Path file = directory.resolve("nova.dj");
        model.save(file);
        var loaded = assertInstanceOf(DeepJNova.class, ModelLoader.load(file));
        assertEquals(config, loaded.config());
        assertEquals(model.promptFormat(), loaded.promptFormat());
        assertEquals(model.metadata(), loaded.metadata());
        assertArrayEquals(model.forward(new int[]{1, 2}).data,
                loaded.forward(new int[]{1, 2}).data, 1e-6f);
    }
}
