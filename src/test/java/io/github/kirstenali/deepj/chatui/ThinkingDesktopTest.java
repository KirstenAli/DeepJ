package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class ThinkingDesktopTest extends ChatDesktopTestSupport {
    @Test void initialThinkingHeadingKeepsItsFontAndHeightWhenReasoningStarts() throws Exception {
        var fixture = open(new ThinkingService());
        verify(fixture, view -> {
            script(view, "element('thinkingMode').checked=true; startResponse('Check', 1)");
            double font = styleNumber(view, ".thinking-panel summary", "fontSize");
            double height = styleNumber(view, ".thinking-panel summary", "height");
            script(view, "window.initialThinkingPanel = document.querySelector('.thinking-panel')");
            script(view, "updateResponse(1, '<thi'); updateResponse(1, '<think>Check first')");
            assertEquals(13, font);
            assertEquals(font, styleNumber(view, ".thinking-panel summary", "fontSize"));
            assertEquals(height, styleNumber(view, ".thinking-panel summary", "height"));
            assertEquals(true, script(view, "initialThinkingPanel === document.querySelector('.thinking-panel')"));
            script(view, "element('thinkingMode').checked=false; startResponse('Hello', 2)");
            assertEquals(font, styleNumber(view, ".generation-status", "fontSize"));
            assertEquals(height, styleNumber(view, ".generation-status", "height"));
        });
    }

    @Test void thinkingSwitchReachesGenerationAndKeepsTheExistingControlStyle() throws Exception {
        var service = new ThinkingService();
        var fixture = open(service);
        verifyWithoutClosing(fixture, view -> {
            script(view, "setModel('Qwen3'); setAgentWorkspace('/tmp/workspace'); setThinkingAvailable(true)");
            script(view, "element('agentPicker').click(); element('thinkingMode').click()");
            assertEquals("Thinking on", script(view, "element('agentLabel').textContent"));
            assertEquals(styleNumber(view, "#agentMode", "height"), styleNumber(view, "#thinkingMode", "height"));
        });
        reviewThemes(fixture, "thinking-toggle");
        verifyWithoutClosing(fixture, view -> script(view, "element('prompt').value='Calculate'; submitPrompt()"));
        awaitScript(fixture, "!element('prompt').disabled && document.querySelector('.thinking-answer') !== null");
        assertTrue(service.settings.thinking());
        assertTrue(service.settings.withPenalty().thinking());
        assertTrue(service.settings.withNewSeed().thinking());
        reviewThemes(fixture, "thinking-answer");
        verify(fixture, view -> assertEquals("42",
                script(view, "document.querySelector('.thinking-answer').textContent")));
    }

    @Test void agentReasoningAcrossToolCallsUsesTheSamePanelAndStyles() throws Exception {
        var service = new ThinkingService();
        service.agent = true;
        var fixture = open(service);
        verifyWithoutClosing(fixture, view -> {
            script(view, "setModel('Qwen3'); setAgentWorkspace('/tmp/workspace'); setThinkingAvailable(true)");
            assertEquals("Chat options", script(view, "element('agentTitle').textContent"));
            script(view, "element('agentMode').checked=true; element('thinkingMode').click()");
            assertEquals("Agent + thinking", script(view, "element('agentLabel').textContent"));
            script(view, "element('prompt').value='Check'; submitPrompt()");
        });
        awaitScript(fixture, "!element('prompt').disabled && document.querySelector('.thinking-answer') !== null");
        reviewThemes(fixture, "agent-thinking-answer");
        verify(fixture, ThinkingDesktopTest::verifyAgentAnswer);
    }

    private static void verifyAgentAnswer(javafx.scene.web.WebView view) {
        assertEquals("Check first\n\nExplain the result",
                script(view, "document.querySelector('.thinking-content').textContent"));
        assertEquals("Done", script(view, "document.querySelector('.thinking-answer').textContent"));
        assertEquals("Thought process", script(view, "document.querySelector('.thinking-panel summary').textContent"));
        assertEquals(false, script(view, "document.querySelector('.thinking-panel').open"));
    }

    private static void reviewThemes(Fixture fixture, String name) throws Exception {
        for (String theme : List.of("light", "dark")) {
            verifyWithoutClosing(fixture, view -> script(view, "setTheme('" + theme + "')"));
            ImageDesktopSnapshots.capture(fixture, name + "-" + theme);
        }
    }

    private static final class ThinkingService implements ConversationChatService {
        volatile ChatGenerationSettings settings;
        boolean agent;
        @Override public void loadModel(Path path) {}
        @Override public boolean isModelLoaded() { return true; }
        @Override public String getLoadedModelName() { return "Qwen3"; }
        @Override public boolean supportsThinking() { return true; }
        @Override public String generate(String text, int max, float temp, int top, long seed,
                                         Consumer<String> stream) {
            return "unused";
        }
        @Override public Agent.Turn generateTurn(String prompt, List<Agent.Turn> history,
                                                 ChatGenerationSettings options, Consumer<String> stream) {
            settings = options;
            if (agent) return agentTurn(prompt, stream);
            stream.accept("<think>Calculate carefully");
            return Agent.Turn.answered(prompt, "<think>Calculate carefully</think>42");
        }

        private static Agent.Turn agentTurn(String prompt, Consumer<String> stream) {
            var thinking = new AgentThinking(stream);
            var runner = new Agent(messages -> messages.size() == 2
                    ? "<think>Check first</think><tool_call>{\"name\":\"missing\",\"arguments\":{}}</tool_call>"
                    : "<think>Explain the result</think>Done", List.of(), 1);
            try {
                return thinking.finish(runner.runTurn(List.of(), prompt, thinking::answer, thinking::reasoning));
            } catch (InterruptedException stopped) {
                Thread.currentThread().interrupt();
                throw new java.util.concurrent.CancellationException("Agent stopped");
            }
        }
    }
}
