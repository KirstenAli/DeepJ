package io.github.kirstenali.deepj.nn.transformer.qwen3;

import io.github.kirstenali.deepj.agent.Qwen3AgentModel;
import io.github.kirstenali.deepj.nn.transformer.ContextWindowMode;
import io.github.kirstenali.deepj.tokenizers.qwen3.Qwen3Chat;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class Qwen3ThinkingTest {
    @Test void modeUsesTheExistingTemplateAndSharesTheLoadedModel(@TempDir Path directory) throws Exception {
        var chat = Qwen3ChatModel.load(Qwen3ChatFixture.create(directory.resolve("qwen")));
        var thinking = chat.withThinking(true);
        assertSame(chat.model(), thinking.model());
        assertSame(chat.tokenizer(), thinking.tokenizer());
        var messages = List.of(new Qwen3Chat.Message(Qwen3Chat.Role.USER, "Hello"));
        assertEquals(Qwen3Chat.format(messages, true, true), fit(thinking, messages));
        assertEquals(Qwen3Chat.format(messages, true, false), fit(chat, messages));
        assertFalse(new Qwen3AgentModel(chat).thinking());
        assertTrue(new Qwen3AgentModel(chat, 8, .6f, 20, 1, 1, ContextWindowMode.REBUILD, true).thinking());
    }

    @Test void previousReasoningIsRemovedFromBothConversationTemplates(@TempDir Path directory) throws Exception {
        var chat = Qwen3ChatModel.load(Qwen3ChatFixture.create(directory.resolve("qwen")));
        var messages = List.of(new Qwen3Chat.Message(Qwen3Chat.Role.USER, "Hi"),
                new Qwen3Chat.Message(Qwen3Chat.Role.ASSISTANT, "<think>Private reasoning</think>Hello"),
                new Qwen3Chat.Message(Qwen3Chat.Role.USER, "Next"));
        String plain = fit(chat.withThinking(true), messages);
        assertTrue(plain.contains("Hello"));
        assertFalse(plain.contains("Private reasoning"));
        assertEquals(plain, agentPrompt(chat, messages));
    }

    private static String fit(Qwen3ChatModel chat, List<Qwen3Chat.Message> messages) {
        return new Qwen3ChatSupport(chat.model().config(), chat.tokenizer(), chat.thinking(),
                chat.model()::openGeneration).fitConversation(messages, 8);
    }

    private static String agentPrompt(Qwen3ChatModel chat, List<Qwen3Chat.Message> messages) throws Exception {
        var agent = new Qwen3AgentModel(chat, 8, .6f, 20, 1, 1, ContextWindowMode.REBUILD, true);
        var input = messages.stream().map(message -> new io.github.kirstenali.deepj.agent.AgentModel.Message(
                io.github.kirstenali.deepj.agent.AgentModel.Role.valueOf(message.role().name()),
                message.content())).toList();
        var encode = Qwen3AgentModel.class.getDeclaredMethod("encode", List.class);
        encode.setAccessible(true);
        Object prompt = encode.invoke(agent, input);
        var text = prompt.getClass().getDeclaredMethod("text");
        text.setAccessible(true);
        return (String) text.invoke(prompt);
    }
}
