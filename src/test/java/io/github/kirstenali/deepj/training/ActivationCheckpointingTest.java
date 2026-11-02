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
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class ActivationCheckpointingTest {

    static Stream<Supplier<DecoderOnlyModel>> models() {
        int vocab = ByteTokenizer.VOCAB_SIZE;
        return Stream.of(
                () -> new DeepJOrigin(new DeepJOriginConfig(vocab, 8, 32, 4, 2, 64), 1L),
                () -> new DeepJOrbit(new DeepJOrbitConfig(vocab, 8, 32, 4, 2, DeepJOrbitConfig.defaultDFF(32)), 1L),
                () -> new DeepJPrism(new DeepJPrismConfig(vocab, 8, 32, 4, 2, 64, 16, 8), 1L));
    }

    @ParameterizedTest
    @MethodSource("models")
    void checkpointedStepMatchesPlainStepExactly(Supplier<DecoderOnlyModel> factory) throws IOException {
        DecoderOnlyModel plain = factory.get(), checkpointed = factory.get();
        checkpointed.setActivationCheckpointing(true);
        assertFalse(plain.activationCheckpointing());
        float plainLoss = CausalLMTraining.trainer(plain, dataset(), 1e-2f).trainStep(2);
        float checkedLoss = CausalLMTraining.trainer(checkpointed, dataset(), 1e-2f).trainStep(2);
        assertEquals(plainLoss, checkedLoss);
        assertSameWeights(plain.parameters(), checkpointed.parameters());
    }

    private static void assertSameWeights(List<Parameter> expected, List<Parameter> actual) {
        assertEquals(expected.size(), actual.size());
        for (int index = 0; index < expected.size(); index++) {
            assertArrayEquals(expected.get(index).value.toFloatArray(), actual.get(index).value.toFloatArray(), 1e-6f);
        }
    }

    private static TextDataset dataset() throws IOException {
        Path tmp = Files.createTempFile("deepj_ckpt", ".txt");
        Files.writeString(tmp, "the quick brown fox jumps over the lazy dog again and again");
        return TextDataset.fromFile(tmp, new ByteTokenizer(), 8, 5L);
    }
}

