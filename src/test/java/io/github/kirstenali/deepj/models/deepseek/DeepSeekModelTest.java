package io.github.kirstenali.deepj.models.deepseek;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static io.github.kirstenali.deepj.models.CausalLMTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

public class DeepSeekModelTest {

    @TempDir
    Path tempDir;

    private DeepSeekConfig cfg;
    private DeepSeekModel model;

    @BeforeEach
    void setUp() {
        cfg = new DeepSeekConfig(
                ByteTokenizer.VOCAB_SIZE,
                16,
                32,
                4,
                2,
                64,
                16,
                8
        );
        model = new DeepSeekModel(cfg, 42L);
    }

    @Test
    void config_rejectsInvalidCommonParams() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(0, 16, 32, 4, 2, 64, 16, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(256, 16, 33, 4, 2, 64, 16, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(256, 16, 6, 2, 2, 64, 4, 2));
    }

    @Test
    void config_rejectsInvalidRanks() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(256, 16, 32, 4, 2, 64, 0, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(256, 16, 32, 4, 2, 64, 16, 0));
    }

    @Test
    void configIncludesStableTrainingDefaults() {
        assertEquals(0.2f, cfg.initScale(), 1e-12f);
        assertEquals(1.0f, cfg.gradClipNorm(), 1e-12f);
        assertThrows(IllegalArgumentException.class,
                () -> new DeepSeekConfig(11, 8, 4, 2, 1, 8, 3, 2, 0.0f, 1.0f));
    }

    @Test
    void modelAppliesInitScaleWithoutScalingNormGains() {
        DeepSeekConfig unscaled = new DeepSeekConfig(11, 8, 4, 2, 1, 8, 3, 2, 1.0f, 1.0f);
        DeepSeekConfig scaled = new DeepSeekConfig(11, 8, 4, 2, 1, 8, 3, 2, 0.2f, 1.0f);
        DeepSeekModel baseModel = new DeepSeekModel(unscaled, 1234L);
        DeepSeekModel scaledModel = new DeepSeekModel(scaled, 1234L);
        float ratio = scaledModel.parameters().get(0).value.sumAbs()
                / baseModel.parameters().get(0).value.sumAbs();
        assertEquals(0.2f, ratio, 1e-6f);
        assertTrue(scaledModel.parameters().stream().anyMatch(p -> isAllOnes(p.value)));
    }

    @Test
    void forward_producesLogitsOfShape_seqLenByVocab() {
        assertLogitShape(model, cfg, new int[]{ 1, 2, 3, 4 });
    }

    @Test
    void forward_singleToken_doesNotThrow() {
        assertLogitShape(model, cfg, new int[]{ 5 });
    }

    @Test
    void forward_fullContextWindow_doesNotThrow() {
        assertFullContext(model, cfg);
    }

    @Test
    void backward_accumulatesGradients() {
        assertBackward(model);
    }

    @Test
    void parameters_countMatchesExpectedStructure() {
        assertParameterCount(model, cfg, 14);
    }

    @Test
    void gradClipNorm_matchesConfig() {
        assertTrainingConfig(model, cfg, model.config());
    }

    @Test
    void checkpointRoundTripPreservesLogits() throws IOException {
        assertCheckpoint(model, new DeepSeekModel(cfg, 99L), tempDir.resolve("deepseek.dj"));
    }

    @Test
    void generate_runsAndStartsWithPrompt() {
        assertGeneration(model, cfg);
    }

    @Test
    void generate_sameSeedProducesSameOutput() {
        assertRepeatableGeneration(model, cfg);
    }
}
