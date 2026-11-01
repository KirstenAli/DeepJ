package io.github.kirstenali.deepj.publishing;

import io.github.kirstenali.deepj.nn.transformer.*;
import io.github.kirstenali.deepj.nn.transformer.origin.*;
import io.github.kirstenali.deepj.nn.transformer.prism.*;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.bpe.*;
import java.io.IOException;
import java.nio.file.*;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

class TypedModelBundleTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }
    static Stream<Arguments> cases() {
        return Stream.of("origin", "prism").flatMap(family -> Stream.of(DType.values())
                .map(dtype -> Arguments.of(family, dtype)));
    }

    @ParameterizedTest
    @MethodSource("cases")
    void exportsStateItsActualCheckpointFormatAndRetainTypedStorage(String family, DType dtype) throws IOException {
        BPEModel tokenizer = new BPETrainer().trainTokenizerWithDefaults("hello world", 280).model();
        int vocabulary = tokenizer.vocabSize();
        DecoderOnlyModel model = family.equals("origin")
                ? new DeepJOrigin(new DeepJOriginConfig(vocabulary, 16, 8, 2, 1, 16), 42, dtype)
                : new DeepJPrism(new DeepJPrismConfig(vocabulary, 16, 8, 2, 1, 16, 4, 2), 42, dtype);
        export(model, tokenizer);
        int version = ModelSerializer.formatVersion(directory.resolve("model.dj"));
        String config = Files.readString(directory.resolve("config.json"));
        assertTrue(config.contains("\"checkpoint_format_version\": " + version));
        assertEquals(dtype, ModelLoader.load(directory.resolve("model.dj")).parameters().get(0).value.dtype());
        String card = Files.readString(directory.resolve("README.md"));
        if (dtype != DType.FLOAT32) {
            assertTrue(card.contains("typed checkpoint format 5"));
            assertTrue(card.contains("<version>0.12.0-alpha</version>"));
        }
    }

    private void export(DecoderOnlyModel model, BPEModel tokenizer) throws IOException {
        var card = new ModelCard("Typed fixture", "Test bundle", "mit", "en", "Synthetic", "Experimental");
        if (model instanceof DeepJOrigin origin) DeepJModelBundle.export(directory, origin, tokenizer, card);
        else DeepJModelBundle.export(directory, (DeepJPrism) model, tokenizer, card);
    }
}
