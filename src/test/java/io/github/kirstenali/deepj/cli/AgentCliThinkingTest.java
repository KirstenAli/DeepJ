package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.agent.AgentModel;
import io.github.kirstenali.deepj.agent.InteractiveTerminal;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class AgentCliThinkingTest extends CliTestSupport {
    @TempDir Path directory;

    @Test void agentThinkingStreamsAcrossApprovedToolCallsAndCanBeToggled() {
        assumeTrue(InteractiveTerminal.available());
        console.inputs.addAll(List.of("Where?", "yes", "/set history on", "/set thinking off", "Again",
                "/set thinking on", "/settings", "/help", "/exit"));
        var modes = new ArrayList<Boolean>();
        new AgentCliSession(console, view(), directory, enabled -> {
            modes.add(enabled);
            return scripted(enabled);
        }, true).run();
        assertEquals(List.of(true, false, true), modes);
        String text = output.toString();
        assertTrue(text.indexOf("Thinking") >= 0);
        assertTrue(text.indexOf("Thinking") < text.indexOf("Command and arguments:"));
        assertTrue(text.contains("Check the directory") && text.contains("Read the tool result"));
        assertTrue(text.contains("Done") && text.contains("Plain reply"));
        assertFalse(text.contains("<think>") || text.contains("<tool_call>"));
    }

    @Test void unsupportedAndInvalidTogglesDoNotReplaceTheActiveAgent() {
        assumeTrue(InteractiveTerminal.available());
        console.inputs.addAll(List.of("/set thinking on", "/set thinking maybe", "/set", "/set unknown on",
                "Hello", "/exit"));
        new AgentCliSession(console, view(), messages -> "Still working", directory).run();
        assertTrue(output.toString().contains("Thinking is unavailable"));
        assertTrue(output.toString().contains("Still working"));
    }

    @Test void reasoningStepsCompleteOnceAndUnfinishedTextEndsOnItsOwnLine() {
        try (var reasoning = new CliReasoning(view())) {
            reasoning.accept(new Agent.Reasoning(0, "One", false));
            reasoning.accept(new Agent.Reasoning(0, "One 😀", true));
            reasoning.accept(new Agent.Reasoning(0, "One 😀", true));
            reasoning.accept(new Agent.Reasoning(1, "Two", false));
        }
        assertEquals(2, output.toString().split("Thinking", -1).length - 1);
        assertTrue(output.toString().endsWith("Two" + System.lineSeparator()));
    }

    private CliView view() { return new CliView(console, new CliOptions(), Map.of("NO_COLOR", "1")); }

    private static AgentModel scripted(boolean thinking) {
        return messages -> {
            if (!thinking) {
                assertTrue(messages.stream().noneMatch(message -> message.content().contains("<think>")));
                return "Plain reply";
            }
            if (messages.size() == 2) return """
                    <think>Check the directory</think>
                    <tool_call>{"name":"terminal","arguments":{"argv":["/bin/pwd"]}}</tool_call>
                    """;
            verifyToolResult(messages.get(3).content());
            return "<think>Read the tool result</think>Done";
        };
    }

    private static void verifyToolResult(String text) {
        String json = text.lines().skip(1).findFirst().orElseThrow();
        var response = Qwen3Json.object(Qwen3Json.parse(json));
        var result = Qwen3Json.object(response.get("result"));
        assertEquals(0, ((Number) result.get("exitCode")).intValue(), text);
    }
}
