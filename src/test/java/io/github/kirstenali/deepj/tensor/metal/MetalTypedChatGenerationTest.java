package io.github.kirstenali.deepj.tensor.metal;

import io.github.kirstenali.deepj.chatui.DeepJChatService;
import io.github.kirstenali.deepj.models.*;
import io.github.kirstenali.deepj.tensor.*;
import io.github.kirstenali.deepj.testing.TypedChatFixtures;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import java.nio.file.*;
import java.util.ArrayList;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import static org.junit.jupiter.api.Assertions.*;

class MetalTypedChatGenerationTest extends MetalDifferentialTestSupport {
    @TempDir Path directory;

    @ParameterizedTest
    @MethodSource("io.github.kirstenali.deepj.testing.NdModelFixtures#typedCases")
    void loadedTypedModelsStreamThroughThePublicChatApiAndDesktopServiceOnMetal(String family, DType dtype)
            throws Exception {
        Path file = TypedChatFixtures.save(family, dtype, directory);
        DecoderOnlyModel model = ModelLoader.load(file);
        String formatted = model.promptFormat().apply("Hello");
        String expected = TextGenerator.generate(model, new ByteTokenizer(), model.config(), formatted,
                4, 0.8f, 40, 1234, 1.1f).substring(formatted.length());
        DeepJChat chat = DeepJChat.load(file);
        var updates = new ArrayList<String>();
        assertEquals(expected, chat.stream("Hello", 4, 0.8f, 40, 1234, 1.1f, updates::add));
        assertEquals(expected, updates.get(updates.size() - 1));
        DeepJChatService service = new DeepJChatService();
        service.loadModel(file);
        assertEquals(expected, service.generate("Hello", 4, 0.8f, 40, 1234, 1.1f, text -> {}));
        assertInstanceOf(MetalBackend.class, Tensor.backend());
    }
}
