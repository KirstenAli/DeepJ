package io.github.kirstenali.deepj.publishing;

import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.ModelLoader;
import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
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
                new DeepJOrigin(config, 42L), tokenizer, card());

        assertTrue(Files.isRegularFile(bundle.resolve(DeepJModelBundle.MODEL_FILE)));
        BPEModel restoredTokenizer = BPEModelIO.load(bundle.resolve(DeepJModelBundle.TOKENIZER_FILE));
        assertEquals(tokenizer.vocabSize(), restoredTokenizer.vocabSize());
        assertDoesNotThrow(() -> new DeepJOrigin(config, 1L).load(bundle.resolve(DeepJModelBundle.MODEL_FILE)));
        assertEquals(config, ModelLoader.load(bundle.resolve(DeepJModelBundle.MODEL_FILE)).config());
        assertTrue(Files.readString(bundle.resolve(DeepJModelBundle.CONFIG_FILE)).contains("\"deepj-origin\""));
        assertTrue(Files.readString(bundle.resolve(DeepJModelBundle.MODEL_CARD_FILE)).contains("library_name: deepj"));
    }

    @Test
    void exportRejectsTokenizerAndConfigVocabularyMismatch() {
        BPEModel tokenizer = tokenizer();
        DeepJOriginConfig wrongConfig = config(tokenizer.vocabSize() + 1);
        assertThrows(IllegalArgumentException.class, () -> DeepJModelBundle.export(
                temporaryDirectory, new DeepJOrigin(wrongConfig, 42L), tokenizer, card()));
    }

    @Test
    void exportSupportsDeepJPrism() throws Exception {
        BPEModel tokenizer = tokenizer();
        DeepJPrismConfig config = new DeepJPrismConfig(tokenizer.vocabSize(), 16, 8, 2, 1, 16, 4, 2);
        DeepJPrism model = new DeepJPrism(config, 42L);
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        Path bundle = DeepJModelBundle.export(temporaryDirectory.resolve("prism-bundle"),
                model, tokenizer, card());
        String json = Files.readString(bundle.resolve(DeepJModelBundle.CONFIG_FILE));
        String modelCard = Files.readString(bundle.resolve(DeepJModelBundle.MODEL_CARD_FILE));
        assertDeepJPrismConfig(json);
        assertTrue(json.contains("\"prompt_format\": \"instruction_response\""));
        assertDeepJPrismCard(modelCard);
        assertDoesNotThrow(() -> new DeepJPrism(config, 1L)
                .load(bundle.resolve(DeepJModelBundle.MODEL_FILE)));
        assertEquals(PromptFormat.INSTRUCTION_RESPONSE,
                ModelLoader.load(bundle.resolve(DeepJModelBundle.MODEL_FILE)).promptFormat());
    }

    @Test
    void olderExportLoadsFromItsConfigJson() throws Exception {
        BPEModel tokenizer = tokenizer();
        DeepJOriginConfig config = config(tokenizer.vocabSize());
        DeepJOrigin model = new DeepJOrigin(config, 42L);
        Path bundle = DeepJModelBundle.export(temporaryDirectory.resolve("legacy"), model, tokenizer, card());
        Path checkpoint = bundle.resolve(DeepJModelBundle.MODEL_FILE);
        ModelSerializer.save(model.parameters(), checkpoint);
        assertEquals(config, ModelLoader.load(checkpoint).config());
    }

    private static void assertDeepJPrismConfig(String json) {
        assertTrue(json.contains("\"deepj-prism\""));
        assertTrue(json.contains("\"q_rank\": 4"));
    }

    private static void assertDeepJPrismCard(String modelCard) {
        assertTrue(modelCard.contains("created with [DeepJ](https://github.com/KirstenAli/DeepJ)"));
        assertTrue(modelCard.contains("DeepJ Prism Transformer architecture"));
        assertTrue(modelCard.contains("a hidden size of 8"));
        assertTrue(modelCard.contains("vocabulary.\n\nDeepJ Prism uses low-rank Q/KV attention"));
        assertTrue(modelCard.contains("recalculates the full context for every generated token"));
        assertTrue(modelCard.contains("## Usage"));
        assertTrue(modelCard.contains("checkpoint format 2"));
        assertFalse(modelCard.contains("tree/deepj-0.6-tinystories-release"));
        assertTrue(modelCard.contains("ModelLoader.load(directory.resolve(\"model.dj\"))"));
        assertTrue(modelCard.contains("model.promptFormat().apply(\"Once upon a time\")"));
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
