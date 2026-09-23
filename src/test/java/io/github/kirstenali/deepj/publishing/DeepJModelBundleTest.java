package io.github.kirstenali.deepj.publishing;

import io.github.kirstenali.deepj.models.deepseek.DeepSeekConfig;
import io.github.kirstenali.deepj.models.deepseek.DeepSeekModel;
import io.github.kirstenali.deepj.models.gpt.GPTConfig;
import io.github.kirstenali.deepj.models.gpt.GPTModel;
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
        GPTConfig config = config(tokenizer.vocabSize());
        Path bundle = DeepJModelBundle.export(temporaryDirectory.resolve("bundle"),
                new GPTModel(config, 42L), tokenizer, card());

        assertTrue(Files.isRegularFile(bundle.resolve(DeepJModelBundle.MODEL_FILE)));
        BPEModel restoredTokenizer = BPEModelIO.load(bundle.resolve(DeepJModelBundle.TOKENIZER_FILE));
        assertEquals(tokenizer.vocabSize(), restoredTokenizer.vocabSize());
        assertDoesNotThrow(() -> new GPTModel(config, 1L).load(bundle.resolve(DeepJModelBundle.MODEL_FILE)));
        assertTrue(Files.readString(bundle.resolve(DeepJModelBundle.CONFIG_FILE)).contains("\"deepj-gpt\""));
        assertTrue(Files.readString(bundle.resolve(DeepJModelBundle.MODEL_CARD_FILE)).contains("library_name: deepj"));
    }

    @Test
    void exportRejectsTokenizerAndConfigVocabularyMismatch() {
        BPEModel tokenizer = tokenizer();
        GPTConfig wrongConfig = config(tokenizer.vocabSize() + 1);
        assertThrows(IllegalArgumentException.class, () -> DeepJModelBundle.export(
                temporaryDirectory, new GPTModel(wrongConfig, 42L), tokenizer, card()));
    }

    @Test
    void exportSupportsDeepSeekStyleModels() throws Exception {
        BPEModel tokenizer = tokenizer();
        DeepSeekConfig config = new DeepSeekConfig(tokenizer.vocabSize(), 16, 8, 2, 1, 16, 4, 2);
        Path bundle = DeepJModelBundle.export(temporaryDirectory.resolve("deepseek-bundle"),
                new DeepSeekModel(config, 42L), tokenizer, card());
        String json = Files.readString(bundle.resolve(DeepJModelBundle.CONFIG_FILE));
        String modelCard = Files.readString(bundle.resolve(DeepJModelBundle.MODEL_CARD_FILE));
        assertDeepSeekConfig(json);
        assertDeepSeekModelCard(modelCard);
        assertDoesNotThrow(() -> new DeepSeekModel(config, 1L)
                .load(bundle.resolve(DeepJModelBundle.MODEL_FILE)));
    }

    private static void assertDeepSeekConfig(String json) {
        assertTrue(json.contains("\"deepj-deepseek-style\""));
        assertTrue(json.contains("\"q_rank\": 4"));
    }

    private static void assertDeepSeekModelCard(String modelCard) {
        assertTrue(modelCard.contains("created with [DeepJ](https://github.com/KirstenAli/DeepJ)"));
        assertTrue(modelCard.contains("DeepSeek-style Transformer architecture"));
        assertTrue(modelCard.contains("a hidden size of 8"));
        assertTrue(modelCard.contains("vocabulary.\n\nIt is not an exact implementation"));
        assertTrue(modelCard.contains("not an exact implementation of DeepSeek V2, V3 or R1"));
        assertTrue(modelCard.contains("recalculates the full context for every generated token"));
        assertTrue(modelCard.contains("## Usage"));
        assertTrue(modelCard.contains("io.github.kirstenali"));
        assertTrue(modelCard.contains("<version>0.7.2-alpha</version>"));
        assertFalse(modelCard.contains("tree/deepj-0.6-tinystories-release"));
        assertTrue(modelCard.contains("model.load(directory.resolve(\"model.dj\"))"));
        assertTrue(modelCard.contains("0.2f, 1.0f"));
    }

    private static BPEModel tokenizer() {
        return new BPETrainer().trainTokenizerWithDefaults(
                "hello world hello deepj world", 280).model();
    }

    private static GPTConfig config(int vocabSize) {
        return new GPTConfig(vocabSize, 16, 8, 2, 1, 16);
    }

    private static ModelCard card() {
        return new ModelCard("DeepJ test model", "A small test model.", "mit", "en",
                "Synthetic unit-test text.", "Not intended for real inference.");
    }
}
