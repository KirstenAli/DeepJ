package io.github.kirstenali.deepj.agent;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AgentReasoningTest {
    @Test void streamsReasoningAcrossToolCallsWithoutMixingItIntoAnswersOrHistory() throws InterruptedException {
        var thoughts = new ArrayList<Agent.Reasoning>();
        var answers = new ArrayList<String>();
        var agent = new Agent(streamingModel(), List.of(), 1);
        var turn = agent.runTurn(List.of(), "Check", answers::add, thoughts::add);
        assertEquals("Done", turn.reply());
        assertEquals(List.of("Do", "Done"), answers);
        assertTrue(thoughts.contains(new Agent.Reasoning(0, "Check the directory", false)));
        assertTrue(thoughts.contains(new Agent.Reasoning(0, "Check the directory", true)));
        assertTrue(thoughts.contains(new Agent.Reasoning(1, "Use the result", false)));
        assertEquals(new Agent.Reasoning(1, "Use the result", true), thoughts.get(thoughts.size() - 1));
        assertTrue(turn.messages().stream().noneMatch(message -> message.content().contains("<think>")));
        assertTrue(turn.messages().stream().anyMatch(message -> message.role() == AgentModel.Role.TOOL));
    }

    @Test void reportsFinalReasoningWhenAModelDoesNotStreamAndThereIsNoAnswerListener()
            throws InterruptedException {
        var thoughts = new ArrayList<Agent.Reasoning>();
        var agent = new Agent(messages -> "<think>Consider this</think>Answer", List.of(), 0);
        var turn = agent.runTurn(List.of(), "Test", null, thoughts::add);
        assertEquals("Answer", turn.reply());
        assertEquals(List.of(new Agent.Reasoning(0, "Consider this", true)), thoughts);
    }

    @Test void ordinaryAnswersDoNotCreateReasoningEvents() throws InterruptedException {
        var agent = new Agent(messages -> "Answer", List.of(), 0);
        var turn = agent.runTurn(List.of(), "Test", null, progress -> fail("Unexpected reasoning"));
        assertEquals("Answer", turn.reply());
    }

    @Test void incompleteReasoningNeverBecomesAnExecutableToolCall() {
        var thoughts = new ArrayList<Agent.Reasoning>();
        var agent = new Agent(messages -> "<think>Consider <tool_call>{", List.of(), 1);
        assertThrows(IllegalArgumentException.class,
                () -> agent.runTurn(List.of(), "Test", null, thoughts::add));
        assertEquals(List.of(new Agent.Reasoning(0, "Consider <tool_call>{", false)), thoughts);
    }

    private static AgentModel streamingModel() {
        return new AgentModel() {
            @Override public String generate(List<Message> messages) { return fail("Expected streaming"); }
            @Override public String stream(List<Message> messages, Consumer<String> listener) {
                var parts = messages.size() == 2 ? beforeTool() : afterTool();
                parts.forEach(listener);
                return parts.get(parts.size() - 1);
            }
        };
    }

    private static List<String> beforeTool() {
        String reasoning = "  <think>Check the directory";
        return List.of("  <thi", reasoning, reasoning + "</think><tool_call>", reasoning
                + "</think><tool_call>{\"name\":\"missing\",\"arguments\":{}}</tool_call>");
    }

    private static List<String> afterTool() {
        return List.of("<think>Use the result", "<think>Use the result</think>Do",
                "<think>Use the result</think>Done");
    }
}
