package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.transformer.TextGenerator;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrism;
import io.github.kirstenali.deepj.nn.transformer.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.TensorBackend;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tensor.metal.MetalBackend;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class ChatBackendTest {

    @Test
    void selectsAvailableBackendBeforeLoadingModels() {
        TensorBackend previous = Tensor.backend();
        try {
            ChatBackend.configure();
            Class<? extends TensorBackend> expected = MetalBackend.isAvailable()
                    ? MetalBackend.class : CpuBackend.class;
            assertInstanceOf(expected, Tensor.backend());
            assertEquals(MetalBackend.isAvailable() ? "Ready on Metal" : "Ready on CPU",
                    ChatBackend.readyStatus());
        } finally {
            Tensor.setBackend(previous);
        }
    }

    @Test
    void repeatedStreamingKeepsTheSelectedBackendWorking() {
        TensorBackend previous = Tensor.backend();
        try {
            ChatBackend.configure();
            var config = new DeepJPrismConfig(ByteTokenizer.VOCAB_SIZE, 16, 16, 4, 1, 32, 8, 8);
            var model = new DeepJPrism(config, 42L);
            var tokenizer = new ByteTokenizer();
            String first = TextGenerator.stream(model, tokenizer, config, "Hello", 4, 1.0f, 5,
                    7L, ignored -> {});
            String second = TextGenerator.stream(model, tokenizer, config, "Hello", 4, 1.0f, 5,
                    7L, ignored -> {});
            assertEquals(first, second);
        } finally {
            Tensor.backend().releaseResources();
            Tensor.setBackend(previous);
        }
    }
}
