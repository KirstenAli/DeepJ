package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETokenizer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeepJChatTest {

    @TempDir Path directory;

    @Test
    void loadsBundleAndReturnsOnlyGeneratedResponse() throws IOException {
        BPEModel bpe = tokenizer();
        DeepJPrism model = savePrism(bpe.vocabSize());
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), bpe);
        BPETokenizer tokenizer = new BPETokenizer(bpe);
        String prompt = model.promptFormat().apply("Hello");
        String full = TextGenerator.generate(model, tokenizer, model.config(),
                prompt, 3, 0.8f, 40, 1234L);
        String prefix = tokenizer.decode(tokenizer.encode(prompt));
        assertEquals(full.substring(prefix.length()), DeepJChat.load(directory)
                .generate("Hello", 3, 0.8f, 40, 1234L));
    }

    @Test
    void streamingMatchesGeneration() throws IOException {
        BPEModel bpe = tokenizer();
        savePrism(bpe.vocabSize());
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), bpe);
        DeepJChat chat = DeepJChat.load(directory);
        List<String> updates = new ArrayList<>();
        String streamed = chat.stream("Hello", 3, 0.8f, 40, 1234L, updates::add);
        assertEquals(chat.generate("Hello", 3, 0.8f, 40, 1234L), streamed);
        assertEquals(streamed, updates.get(updates.size() - 1));
    }

    @Test
    void simpleCallUsesDesktopDefaults() throws IOException {
        BPEModel bpe = tokenizer();
        savePrism(bpe.vocabSize());
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), bpe);
        DeepJChat chat = DeepJChat.load(directory);
        assertEquals(chat.generate("Hello", 80, 0.8f, 40, 1234L), chat.generate("Hello"));
    }

    @Test
    void customRepetitionPenaltyMatchesLowLevelGeneration() throws IOException {
        BPEModel bpe = tokenizer();
        DeepJPrism model = savePrism(bpe.vocabSize());
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), bpe);
        BPETokenizer tokenizer = new BPETokenizer(bpe);
        String prompt = model.promptFormat().apply("Hello");
        String prefix = tokenizer.decode(tokenizer.encode(prompt));
        String full = TextGenerator.generate(model::forward, model.config().maxSeqLen(),
                tokenizer, prompt, 3, 0.8f, 40, 1234L, 1.2f);
        DeepJChat chat = DeepJChat.load(directory);
        String expected = full.substring(prefix.length());
        assertEquals(expected, chat.generate("Hello", 3, 0.8f, 40, 1234L, 1.2f));
        assertEquals(expected, chat.stream("Hello", 3, 0.8f, 40, 1234L, 1.2f, text -> {}));
    }

    @Test
    void supportsByteTokenizerWhenNoBpeFileExists() throws IOException {
        savePrism(256);
        assertDoesNotThrow(() -> DeepJChat.load(directory));
    }

    @Test
    void rejectsTokenizerVocabularyMismatch() throws IOException {
        BPEModel bpe = tokenizer();
        savePrism(bpe.vocabSize() + 1);
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), bpe);
        IOException error = assertThrows(IOException.class, () -> DeepJChat.load(directory));
        assertTrue(error.getMessage().contains("vocabulary"));
    }

    @Test
    void reportsMissingBpeTokenizer() throws IOException {
        savePrism(280);
        IOException error = assertThrows(IOException.class, () -> DeepJChat.load(directory));
        assertTrue(error.getMessage().contains("tokenizer.bpe"));
    }

    private DeepJPrism savePrism(int vocabSize) throws IOException {
        var config = new DeepJPrismConfig(vocabSize, 32, 8, 2, 1, 16, 4, 2);
        var model = new DeepJPrism(config, 42L);
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        model.save(directory.resolve("model.dj"));
        return model;
    }

    private static BPEModel tokenizer() {
        return new BPETrainer().trainTokenizerWithDefaults("hello world hello deepj world", 280).model();
    }
}
