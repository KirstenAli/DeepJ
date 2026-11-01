package io.github.kirstenali.deepj.nn.transformer;

import io.github.kirstenali.deepj.chatui.DeepJChatService;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.testing.TypedChatFixtures;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import java.nio.file.*;
import java.util.ArrayList;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

class TypedChatGenerationTest {
    @TempDir Path directory;
    @BeforeEach void cpu() { Tensor.setBackend(new CpuBackend()); }

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void publicApiAndDesktopServiceStreamTypedCheckpointsWithTheExactPromptFormat(String family, DType dtype)
            throws Exception {
        Path file = TypedChatFixtures.save(family, dtype, directory);
        DecoderOnlyModel model = ModelLoader.load(file);
        String formatted = model.promptFormat().apply("Hello");
        String full = TextGenerator.generate(model, new ByteTokenizer(), model.config(), formatted,
                4, 0.8f, 40, 1234, 1.1f);
        String expected = full.substring(formatted.length());
        var updates = new ArrayList<String>();
        DeepJChat chat = DeepJChat.load(file);
        assertEquals(expected, chat.generate("Hello", 4, 0.8f, 40, 1234, 1.1f));
        assertEquals(expected, chat.stream("Hello", 4, 0.8f, 40, 1234, 1.1f, updates::add));
        assertEquals(expected, updates.get(updates.size() - 1));
        compareService(file, expected);
    }

    private static void compareService(Path file, String expected) throws Exception {
        DeepJChatService service = new DeepJChatService();
        service.loadModel(file);
        var updates = new ArrayList<String>();
        assertTrue(service.isModelLoaded());
        assertEquals(expected, service.generate("Hello", 4, 0.8f, 40, 1234, 1.1f, updates::add));
        assertEquals(expected, updates.get(updates.size() - 1));
    }

    @ParameterizedTest
    @EnumSource(DType.class)
    void lowLevelSamplingUsesTheLogicalLastRowForNoncontiguousTypedLogits(DType dtype) {
        Tensor logits = new Tensor(dtype, 256, 2).transpose(0, 1);
        logits.set(0, 65, 9);
        logits.set(1, 72, 10);
        assertEquals("AHHH", TextGenerator.generate(context -> logits, 32,
                new ByteTokenizer(), "A", 3, 0.8f, 1, 1234));
    }
}
