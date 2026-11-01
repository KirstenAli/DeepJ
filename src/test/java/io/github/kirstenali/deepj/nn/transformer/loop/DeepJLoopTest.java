package io.github.kirstenali.deepj.nn.transformer.loop;

import io.github.kirstenali.deepj.data.Batch;
import io.github.kirstenali.deepj.loss.CrossEntropyLoss;
import io.github.kirstenali.deepj.nn.transformer.ModelLoader;
import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.training.CausalLMTraining;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DeepJLoopTest {

    @TempDir Path directory;

    @BeforeEach
    void selectCpu() {
        Tensor.setBackend(new CpuBackend());
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 2, 4})
    void repeatedPassesDoNotDuplicateWeights(int loops) {
        var settings = small(loops);
        var decoder = new DeepJLoop(settings, 17L);
        long weights = decoder.parameters().stream().mapToLong(p -> p.value.rawData().length).sum();
        assertEquals(DeepJLoopParameterCount.count(settings), weights);
        assertEquals(decoder.parameters().size(), decoder.parameters().stream().distinct().count());
        assertEquals(3, settings.nLayers());
        assertEquals(loops + 2, settings.executedLayers());
        assertEquals(1, decoder.parameters().stream().filter(p -> p.value.matrixRows() == 11).count());
    }

    @Test
    void productionConfigurationCountsUniqueParameters() {
        var settings = new DeepJLoopConfig(16_384, 1_024, 576, 9, 3, 6, 12, 6, 2, 1_664);
        assertEquals(100_375_680L, DeepJLoopParameterCount.count(settings));
        assertEquals(24, settings.nLayers());
        assertEquals(36, settings.executedLayers());
        assertEquals(1.0 / (2 * Math.sqrt(12)), settings.coreResidualScale(), 1e-7);
    }

    @Test
    void changingPassCountDoesNotChangeInitialWeights() {
        var once = new DeepJLoop(small(1), 17L);
        var repeated = new DeepJLoop(small(4), 17L);
        for (int index = 0; index < once.parameters().size(); index++) {
            assertArrayEquals(once.parameters().get(index).value.rawData(),
                    repeated.parameters().get(index).value.rawData(), 0.0f);
        }
    }

    @Test
    void futureTokensDoNotAffectEarlierPredictions() {
        var decoder = new DeepJLoop(small(2), 17L);
        float[] first = decoder.forward(new int[]{1, 2, 3}).rawData().clone();
        float[] changed = decoder.forward(new int[]{1, 2, 8}).rawData();
        assertArrayEquals(Arrays.copyOf(first, 22), Arrays.copyOf(changed, 22), 1e-6f);
    }

    @Test
    void repeatedBatchLearnsWithSharedCoreAndEmbedding() {
        var decoder = new DeepJLoop(small(2), 17L);
        int[] tokens = {1, 3, 5}, targets = {3, 5, 1};
        var batch = new Batch(new int[][]{tokens}, new int[][]{targets});
        float initial = CrossEntropyLoss.loss(decoder.forward(tokens), targets);
        var trainer = CausalLMTraining.trainer(decoder, ignored -> batch, 0.01f);
        for (int update = 0; update < 60; update++) {
            assertTrue(Float.isFinite(trainer.trainStep(1)));
        }
        assertTrue(CrossEntropyLoss.loss(decoder.forward(tokens), targets) < initial * 0.6f);
        assertEquals(small(2).gradClipNorm(), decoder.gradClipNorm());
    }

    @Test
    void embeddedSettingsRoundTripPreservesLoopsAndPrompt() throws Exception {
        var decoder = new DeepJLoop(small(3), 17L);
        decoder.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        decoder.setMetadata(Map.of("stage", "sft"));
        Path saved = directory.resolve("loop.dj");
        decoder.save(saved);
        var restored = assertInstanceOf(DeepJLoop.class, ModelLoader.load(saved));
        assertEquals(decoder.config(), restored.config());
        assertEquals(decoder.promptFormat(), restored.promptFormat());
        assertEquals(decoder.metadata(), restored.metadata());
        assertArrayEquals(decoder.forward(new int[]{1, 4}).rawData(),
                restored.forward(new int[]{1, 4}).rawData(), 0.0f);
        assertThrows(java.io.IOException.class, () -> new DeepJLoop(small(2), 17L).load(saved));
    }

    @Test
    void zeroOuterBlocksAndDisabledQkNormRemainSupported() throws Exception {
        var settings = new DeepJLoopConfig(11, 3, 8, 4, 2, 0, 1, 0, 2, 12, false, 0.3f, 0.7f, 0.5f);
        var decoder = new DeepJLoop(settings, 17L);
        assertEquals(DeepJLoopParameterCount.count(settings),
                decoder.parameters().stream().mapToLong(p -> p.value.rawData().length).sum());
        Path saved = directory.resolve("core.dj");
        decoder.save(saved);
        assertEquals(settings, ModelLoader.load(saved).config());
        assertEquals(3, decoder.forward(new int[]{1, 2, 3}).matrixRows());
    }

    private static DeepJLoopConfig small(int loops) {
        return new DeepJLoopConfig(11, 3, 8, 4, 2, 1, 1, 1, loops, 12);
    }
}
