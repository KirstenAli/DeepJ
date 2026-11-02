package io.github.kirstenali.deepj.training;

import io.github.kirstenali.deepj.data.TextDataset;
import io.github.kirstenali.deepj.nn.transformer.DecoderOnlyModel;
import io.github.kirstenali.deepj.nn.transformer.orbit.DeepJOrbit;
import io.github.kirstenali.deepj.nn.transformer.orbit.DeepJOrbitConfig;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOrigin;
import io.github.kirstenali.deepj.nn.transformer.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrism;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.optimisers.Parameter;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/** Recompute must match the saved-graph path on the device, across several optimizer steps. */
final class MetalActivationCheckpointingTest {
    private static MetalBackend gpu;

    @BeforeAll
    static void setUp() {
        Assumptions.assumeTrue(MetalBackend.isAvailable(), "Metal device not available");
        gpu = new MetalBackend();
        Tensor.setBackend(gpu);
    }

    @AfterAll
    static void tearDown() {
        if (gpu != null) gpu.releaseResources();
        Tensor.setBackend(new CpuBackend());
    }

    static Stream<Supplier<DecoderOnlyModel>> models() {
        int vocab = ByteTokenizer.VOCAB_SIZE;
        return Stream.of(
                () -> new DeepJOrigin(new DeepJOriginConfig(vocab, 64, 64, 4, 2, 128), 1L),
                () -> new DeepJOrbit(new DeepJOrbitConfig(vocab, 64, 64, 4, 2, DeepJOrbitConfig.defaultDFF(64)), 1L),
                () -> new DeepJPrism(new DeepJPrismConfig(vocab, 64, 64, 4, 2, 128, 16, 8), 1L));
    }

    @ParameterizedTest
    @MethodSource("models")
    void checkpointedStepsMatchPlainStepsOnDevice(Supplier<DecoderOnlyModel> factory) throws IOException {
        DecoderOnlyModel plain = factory.get(), checkpointed = factory.get();
        checkpointed.setActivationCheckpointing(true);
        Trainer plainTrainer = CausalLMTraining.trainer(plain, dataset(), 1e-3f);
        Trainer checkedTrainer = CausalLMTraining.trainer(checkpointed, dataset(), 1e-3f);
        for (int step = 0; step < 3; step++) {
            assertEquals(plainTrainer.trainStep(2), checkedTrainer.trainStep(2), 1e-5f);
        }
        assertSameWeights(plain.parameters(), checkpointed.parameters());
    }

    private static void assertSameWeights(List<Parameter> expected, List<Parameter> actual) {
        assertEquals(expected.size(), actual.size());
        for (int index = 0; index < expected.size(); index++) {
            assertArrayEquals(expected.get(index).value.toFloatArray(), actual.get(index).value.toFloatArray(), 1e-5f);
        }
    }

    private static TextDataset dataset() throws IOException {
        Path tmp = Files.createTempFile("deepj_metal_ckpt", ".txt");
        Files.writeString(tmp, "the quick brown fox jumps over the lazy dog again and again ".repeat(8));
        return TextDataset.fromFile(tmp, new ByteTokenizer(), 64, 5L);
    }
}

