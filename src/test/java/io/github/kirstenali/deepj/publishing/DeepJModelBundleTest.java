package io.github.kirstenali.deepj.publishing;

import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOriginModel;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrismModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeepJModelBundleTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void exportWritesReloadableBundleAndMetadata() throws Exception {
        BPEModel tokenizer = tokenizer();
        DeepJOriginConfig config = config(tokenizer.vocabSize());
        Path bundle = DeepJModelBundle.export(temporaryDirectory.resolve("bundle"),
                new DeepJOriginModel(config, 42L), tokenizer, card());

        assertTrue(Files.isRegularFile(bundle.resolve(DeepJModelBundle.MODEL_FILE)));
        BPEModel restoredTokenizer = BPEModelIO.load(bundle.resolve(DeepJModelBundle.TOKENIZER_FILE));
        assertEquals(tokenizer.vocabSize(), restoredTokenizer.vocabSize());
        assertDoesNotThrow(() -> new DeepJOriginModel(config, 1L).load(bundle.resolve(DeepJModelBundle.MODEL_FILE)));
        assertTrue(Files.readString(bundle.resolve(DeepJModelBundle.CONFIG_FILE)).contains("\"deepj-origin\""));
        assertTrue(Files.readString(bundle.resolve(DeepJModelBundle.MODEL_CARD_FILE)).contains("library_name: deepj"));
    }

    @Test
    void exportRejectsTokenizerAndConfigVocabularyMismatch() {
        BPEModel tokenizer = tokenizer();
        DeepJOriginConfig wrongConfig = config(tokenizer.vocabSize() + 1);
        assertThrows(IllegalArgumentException.class, () -> DeepJModelBundle.export(
                temporaryDirectory, new DeepJOriginModel(wrongConfig, 42L), tokenizer, card()));
    }

    @Test
    void exportSupportsDeepJPrismModels() throws Exception {
        BPEModel tokenizer = tokenizer();
        DeepJPrismConfig config = new DeepJPrismConfig(tokenizer.vocabSize(), 16, 8, 2, 1, 16, 4, 2);
        Path bundle = DeepJModelBundle.export(temporaryDirectory.resolve("prism-bundle"),
                new DeepJPrismModel(config, 42L), tokenizer, card());
        String json = Files.readString(bundle.resolve(DeepJModelBundle.CONFIG_FILE));
        String modelCard = Files.readString(bundle.resolve(DeepJModelBundle.MODEL_CARD_FILE));
        assertDeepJPrismConfig(json);
        assertDeepJPrismModelCard(modelCard);
        assertDoesNotThrow(() -> new DeepJPrismModel(config, 1L)
                .load(bundle.resolve(DeepJModelBundle.MODEL_FILE)));
    }

    private static void assertDeepJPrismConfig(String json) {
        assertTrue(json.contains("\"deepj-prism\""));
        assertTrue(json.contains("\"q_rank\": 4"));
    }

    private static void assertDeepJPrismModelCard(String modelCard) {
        assertTrue(modelCard.contains("created with [DeepJ](https://github.com/KirstenAli/DeepJ)"));
        assertTrue(modelCard.contains("DeepJ Prism Transformer architecture"));
        assertTrue(modelCard.contains("a hidden size of 8"));
        assertTrue(modelCard.contains("vocabulary.\n\nDeepJ Prism uses low-rank Q/KV attention"));
        assertTrue(modelCard.contains("recalculates the full context for every generated token"));
        assertTrue(modelCard.contains("## Usage"));
        assertTrue(modelCard.contains("io.github.kirstenali"));
        assertTrue(modelCard.contains("<version>0.8.0-alpha</version>"));
        assertFalse(modelCard.contains("tree/deepj-0.6-tinystories-release"));
        assertTrue(modelCard.contains("model.load(directory.resolve(\"model.dj\"))"));
        assertTrue(modelCard.contains("0.2f, 1.0f"));
    }

    private static BPEModel tokenizer() {
        return new BPETrainer().trainTokenizerWithDefaults(
                "hello world hello deepj world", 280).model();
    }

    private static DeepJOriginConfig config(int vocabSize) {
        return new DeepJOriginConfig(vocabSize, 16, 8, 2, 1, 16);
    }

    private static ModelCard card() {
        return new ModelCard("DeepJ test model", "A small test model.", "mit", "en",
                "Synthetic unit-test text.", "Not intended for real inference.");
    }
}
