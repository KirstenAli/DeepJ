package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.PromptFormat;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoop;
import io.github.kirstenali.deepj.nn.transformer.loop.DeepJLoopConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.bpe.BPEModelIO;
import io.github.kirstenali.deepj.tokenizers.bpe.BPETrainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LoopChatServiceTest {

    @TempDir Path directory;

    @Test
    void chatLoadsAndStreamsTheLoopedCheckpointWithItsSavedPromptFormat() throws Exception {
        Tensor.setBackend(new CpuBackend());
        var tokenizer = new BPETrainer().trainTokenizerWithDefaults(
                "Instruction:\nHello\nResponse:\nHello there, how can I help?", 280);
        BPEModelIO.save(directory.resolve("tokenizer.bpe"), tokenizer.model());
        var model = new DeepJLoop(new DeepJLoopConfig(tokenizer.vocabSize(), 64,
                8, 4, 2, 1, 1, 1, 2, 12), 71L);
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        model.save(directory.resolve("model.dj"));
        var service = new DeepJChatService();
        service.loadModel(directory.resolve("model.dj"));
        assertTrue(service.isModelLoaded());
        var updates = new ArrayList<String>();
        String response = service.generate("Hello", 4, 0.8f, 20, 73L, 1.1f, updates::add);
        String expected = ChatModelGeneration.stream(model, tokenizer, "Hello", 4, 0.8f, 20, 73L, 1.1f, text -> {});
        assertEquals(expected, response);
        assertTrue(updates.isEmpty() || response.equals(updates.get(updates.size() - 1)));
    }
}
