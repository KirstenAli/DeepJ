package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrism;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PrismCheckpointTest {

    @TempDir Path directory;

    @Test
    void loadsEmbeddedConfigurationWithoutSidecar() throws IOException {
        DeepJPrism model = model();
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        Path checkpoint = directory.resolve("model.dj");
        model.save(checkpoint);
        assertEquals(PromptFormat.INSTRUCTION_RESPONSE,
                PrismCheckpoint.load(checkpoint).model().promptFormat());
    }

    private DeepJPrism model() throws IOException {
        var bpe = new BPETrainer().trainTokenizerWithDefaults("hello world deepj", 280).model();
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), bpe);
        var config = new DeepJPrismConfig(bpe.vocabSize(), 16, 8, 2, 1, 16, 4, 2);
        return new DeepJPrism(config, 1L);
    }

}
