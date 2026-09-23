package io.github.kirstenali.deepj.models.orbit;

import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static io.github.kirstenali.deepj.models.CausalLMTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

public class DeepJOrbitModelTest {

    @TempDir
    Path temporaryDirectory;

    private DeepJOrbitConfig cfg;
    private DeepJOrbitModel model;

    @BeforeEach
    void setUp() {
        cfg = new DeepJOrbitConfig(
                ByteTokenizer.VOCAB_SIZE,
                16,
                32,
                4,
                2,
                DeepJOrbitConfig.defaultDFF(32)
        );
        model = new DeepJOrbitModel(cfg, 42L);
    }

    @Test
    void config_defaultDFF_isRoundedMultipleOf64() {
        int dFF = DeepJOrbitConfig.defaultDFF(32);
        assertEquals(0, dFF % 64, "defaultDFF must be a multiple of 64");
        assertTrue(dFF > 0, "defaultDFF must be positive");
    }

    @Test
    void configIncludesStableTrainingDefaults() {
        assertEquals(0.2f, cfg.initScale(), 1e-12f);
        assertEquals(1.0f, cfg.gradClipNorm(), 1e-12f);
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(11, 8, 4, 2, 1, 8, 0.0f, 1.0f));
    }

    @Test
    void modelAppliesInitScaleAndUsesIndependentStreams() {
        DeepJOrbitConfig base = new DeepJOrbitConfig(11, 8, 4, 2, 1, 8, 1.0f, 1.0f);
        DeepJOrbitConfig scaled = new DeepJOrbitConfig(11, 8, 4, 2, 1, 8, 0.2f, 1.0f);
        DeepJOrbitModel baseModel = new DeepJOrbitModel(base, 1234L);
        DeepJOrbitModel scaledModel = new DeepJOrbitModel(scaled, 1234L);
        float ratio = scaledModel.parameters().get(0).value.sumAbs()
                / baseModel.parameters().get(0).value.sumAbs();
        assertEquals(0.2f, ratio, 1e-6f);
        assertTrue(scaledModel.parameters().stream().anyMatch(p -> isAllOnes(p.value)));
        assertNotEquals(scaledModel.parameters().get(0).value.data[0],
                scaledModel.parameters().get(3).value.data[0]);
    }

    @Test
    void config_rejectsInvalidParams() {
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(0, 16, 32, 4, 2, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(256, 0, 32, 4, 2, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(256, 16, 0, 4, 2, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(256, 16, 32, 0, 2, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(256, 16, 32, 4, 0, 64));
        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(256, 16, 32, 4, 2, 0));

        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(256, 16, 33, 4, 2, 64));

        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(256, 16, 6, 2, 2, 64));

        assertThrows(IllegalArgumentException.class,
                () -> new DeepJOrbitConfig(256, 16, 32, 4, 2, 64, 0.0f));
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
        assertParameterCount(model, cfg, 12);
    }

    @Test
    void gradClipNorm_matchesConfig() {
        assertTrainingConfig(model, cfg, model.config());
    }

    @Test
    void checkpointRoundTripPreservesLogits() throws IOException {
        assertCheckpoint(model, new DeepJOrbitModel(cfg, 99L), temporaryDirectory.resolve("orbit.dj"));
    }

    @Test
    void generate_runsAndReturnsNonEmptyString() {
        assertGeneration(model, cfg);
    }

    @Test
    void generate_sameSeedProducesSameOutput() {
        assertRepeatableGeneration(model, cfg);
    }
}
