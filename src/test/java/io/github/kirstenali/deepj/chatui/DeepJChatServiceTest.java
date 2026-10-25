package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.ContextWindowMode;

import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.nova.DeepJNova;
import io.github.kirstenali.deepj.models.nova.DeepJNovaConfig;
import io.github.kirstenali.deepj.models.PromptFormat;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.nio.file.Files;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeepJChatServiceTest {

    @TempDir Path directory;

    @Test
    void embeddedServiceStreamsRollingModeAndRejectsAnUnsupportedOrigin() throws Exception {
        var mode = ContextWindowMode.ROLLING;
        Path path = directory.resolve("model.dj");
        new DeepJNova(new DeepJNovaConfig(256, 8, 8, 2, 1, 1, 16), 2).save(path);
        var service = new DeepJChatService();
        assertThrows(IllegalStateException.class, () -> service.generate("Hello", 12, 1, 1, 1, 1, mode, t -> {}));
        service.loadModel(path);
        assertDoesNotThrow(() -> service.generate("Hello", 12, 1, 1, 1, 1, mode, t -> {}));
        new DeepJOrigin(new DeepJOriginConfig(256, 8, 8, 2, 1, 16), 2).save(path);
        service.loadModel(path);
        assertThrows(UnsupportedOperationException.class,
                () -> service.generate("Hello", 12, 1, 1, 1, 1, mode, t -> {}));
    }

    @Test
    void loadsAnExportedOriginCheckpointUsingEmbeddedConfiguration() throws Exception {
        var tokenizer = new BPETrainer().trainTokenizerWithDefaults("hello world deepj", 280).model();
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), tokenizer);
        var config = new DeepJOriginConfig(tokenizer.vocabSize(), 16, 8, 2, 1, 16);
        new DeepJOrigin(config, 1L).save(directory.resolve("model.dj"));
        var service = new DeepJChatService();
        service.loadModel(directory.resolve("model.dj"));
        assertTrue(service.isModelLoaded());
        assertDoesNotThrow(() -> service.generate("Hello", 2, 0.8f, 40, 1L, 1.15f, text -> {}));
    }

    @Test
    void loadsAndStreamsNovaUsingItsEmbeddedSettings() throws Exception {
        var tokenizer = new BPETrainer().trainTokenizerWithDefaults("hello world deepj", 280).model();
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), tokenizer);
        var model = new DeepJNova(new DeepJNovaConfig(tokenizer.vocabSize(), 64, 8, 4, 2, 1, 12), 2L);
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        model.save(directory.resolve("model.dj"));
        var service = new DeepJChatService();
        service.loadModel(directory.resolve("model.dj"));
        assertTrue(service.isModelLoaded());
        assertDoesNotThrow(() -> service.generate("Hello", 2, 0.8f, 40, 1L, 1.1f, text -> {}));
    }

    @Test
    void rejectsLegacyOriginCheckpointEvenWithSidecar() throws Exception {
        var config = new DeepJOriginConfig(256, 16, 8, 2, 1, 16);
        var model = new DeepJOrigin(config, 1L);
        Path checkpoint = directory.resolve("legacy.dj");
        ModelSerializer.save(model.parameters(), checkpoint);
        Files.writeString(directory.resolve("config.json"), legacyConfig());
        var service = new DeepJChatService();
        assertThrows(IOException.class, () -> service.loadModel(checkpoint));
    }

    private static String legacyConfig() {
        return """
                {"model_type":"deepj-origin","vocab_size":256,"max_seq_len":16,
                "d_model":8,"n_heads":2,"n_layers":1,"d_ff":16}
                """;
    }
}
