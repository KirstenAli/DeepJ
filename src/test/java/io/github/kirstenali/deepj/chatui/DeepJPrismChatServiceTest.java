package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.publishing.DeepJModelBundle;
import io.github.kirstenali.deepj.publishing.ModelCard;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModel;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeepJPrismChatServiceTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void loadsBundleAndStreamsGeneratedText() throws Exception {
        Path model = bundle().resolve(DeepJModelBundle.MODEL_FILE);
        DeepJPrismChatService service = new DeepJPrismChatService();
        service.loadModel(model);
        List<String> updates = new ArrayList<>();

        String result = service.generate("hello", 4, 0.7f, 20, 2L, updates::add);

        assertTrue(service.isModelLoaded());
        assertEquals(4, updates.size());
        assertEquals(result, updates.get(updates.size() - 1));
    }

    private Path bundle() throws Exception {
        BPEModel tokenizer = new BPETrainer().train("hello world hello deepj", 280);
        DeepJPrismConfig config = new DeepJPrismConfig(
                tokenizer.vocabSize(), 16, 8, 2, 1, 16, 4, 2);
        return DeepJModelBundle.export(temporaryDirectory.resolve("bundle"),
                new DeepJPrism(config, 42L), tokenizer, card());
    }

    private static ModelCard card() {
        return new ModelCard("Test", "Test model.", "mit", "en",
                "Synthetic text.", "Not for production.");
    }
}
