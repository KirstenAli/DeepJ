package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.persistence.ModelSerializer;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DeepJChatServiceTest {

    @TempDir Path directory;

    @Test
    void loadsAnExportedOriginCheckpointUsingEmbeddedConfiguration() throws Exception {
        var tokenizer = new BPETrainer().trainTokenizerWithDefaults("hello world deepj", 280).model();
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), tokenizer);
        var config = new DeepJOriginConfig(tokenizer.vocabSize(), 16, 8, 2, 1, 16);
        new DeepJOrigin(config, 1L).save(directory.resolve("model.dj"));
        var service = new DeepJChatService();
        service.loadModel(directory.resolve("model.dj"));
        assertTrue(service.isModelLoaded());
    }

    @Test
    void loadsLegacyOriginCheckpointUsingItsSidecar() throws Exception {
        var config = new DeepJOriginConfig(256, 16, 8, 2, 1, 16);
        var model = new DeepJOrigin(config, 1L);
        Path checkpoint = directory.resolve("legacy.dj");
        ModelSerializer.save(model.parameters(), checkpoint);
        Files.writeString(directory.resolve("config.json"), legacyConfig());
        var service = new DeepJChatService();
        service.loadModel(checkpoint);
        assertTrue(service.isModelLoaded());
    }

    private static String legacyConfig() {
        return """
                {"model_type":"deepj-origin","vocab_size":256,"max_seq_len":16,
                "d_model":8,"n_heads":2,"n_layers":1,"d_ff":16}
                """;
    }
}
