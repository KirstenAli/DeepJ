package io.github.kirstenali.deepj.models;

import io.github.kirstenali.deepj.models.origin.DeepJOrigin;
import io.github.kirstenali.deepj.models.origin.DeepJOriginConfig;
import io.github.kirstenali.deepj.models.prism.DeepJPrism;
import io.github.kirstenali.deepj.models.prism.DeepJPrismConfig;
import io.github.kirstenali.deepj.tensor.Tensor;
import io.github.kirstenali.deepj.tensor.cpu.CpuBackend;
import io.github.kirstenali.deepj.tokenizers.ByteTokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

class RollingChatTest {
    @TempDir Path directory;

    @BeforeEach
    void useCpu() { Tensor.setBackend(new CpuBackend()); }

    @Test
    void highLevelChatPreservesTheInstructionTemplateAndMatchesAnExplicitRollingSession() throws IOException {
        DeepJPrism model = save();
        String prompt = "What is a computer?", formatted = model.promptFormat().apply(prompt);
        String expected;
        try (var session = model.openGeneration(ContextWindowMode.ROLLING)) {
            expected = TextGenerator.generate(session, 8, new ByteTokenizer(), formatted, 20, 0.8f, 20, 47, 1.1f);
        }
        var updates = new ArrayList<String>();
        DeepJChat chat = DeepJChat.load(directory).withContextWindow(ContextWindowMode.ROLLING);
        assertEquals(expected.substring(formatted.length()), chat.generate(prompt, 20, 0.8f, 20, 47, 1.1f));
        assertEquals(chat.generate(prompt, 20, 0.8f, 20, 47, 1.1f),
                chat.stream(prompt, 20, 0.8f, 20, 47, 1.1f, updates::add));
        assertEquals(expected.substring(formatted.length()), updates.get(updates.size() - 1));
    }

    @Test
    void selectingRollingDoesNotMutateTheDefaultChatOrSavedCheckpoint() throws IOException {
        save();
        byte[] checkpoint = java.nio.file.Files.readAllBytes(directory.resolve("model.dj"));
        DeepJChat chat = DeepJChat.load(directory);
        String original = chat.generate("Hello", 20, 0.8f, 20, 47);
        chat.withContextWindow(ContextWindowMode.ROLLING).generate("Hello", 20, 0.8f, 20, 47);
        assertEquals(original, chat.generate("Hello", 20, 0.8f, 20, 47));
        assertArrayEquals(checkpoint, java.nio.file.Files.readAllBytes(directory.resolve("model.dj")));
        assertThrows(NullPointerException.class, () -> chat.withContextWindow(null));
    }

    @Test
    void originCannotEnableRollingThroughTheSimpleLoader() throws IOException {
        var model = new DeepJOrigin(new DeepJOriginConfig(256, 8, 8, 2, 1, 16), 47);
        model.save(directory.resolve("model.dj"));
        DeepJChat chat = DeepJChat.load(directory);
        assertThrows(UnsupportedOperationException.class, () -> chat.withContextWindow(ContextWindowMode.ROLLING));
        assertDoesNotThrow(() -> chat.generate("Hi", 2, 0.8f, 20, 47));
    }

    @Test
    void rolledStreamingFailureAndCancellationLeaveTheNextRequestUsable() throws IOException {
        save();
        DeepJChat chat = DeepJChat.load(directory).withContextWindow(ContextWindowMode.ROLLING);
        assertThrows(IllegalStateException.class, () -> chat.stream("Hello", 20, 0.8f, 20, 47, text -> {
            throw new IllegalStateException("listener failed");
        }));
        try {
            assertThrows(java.util.concurrent.CancellationException.class,
                    () -> chat.stream("Hello", 20, 0.8f, 20, 47, text -> Thread.currentThread().interrupt()));
        } finally {
            Thread.interrupted();
        }
        assertDoesNotThrow(() -> chat.generate("Hello", 20, 0.8f, 20, 47));
    }

    private DeepJPrism save() throws IOException {
        var model = new DeepJPrism(new DeepJPrismConfig(256, 8, 8, 2, 1, 16, 4, 2), 47);
        model.setPromptFormat(PromptFormat.INSTRUCTION_RESPONSE);
        model.save(directory.resolve("model.dj"));
        return model;
    }
}
