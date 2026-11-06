package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AgentThinkingTest {
    @Test void keepsReasoningAcrossToolRoundsAndPreservesTheOriginalConversation() throws InterruptedException {
        var updates = new ArrayList<String>();
        var thinking = new AgentThinking(updates::add);
        var agent = new Agent(messages -> messages.size() == 2
                ? "<think>Check first</think><tool_call>{\"name\":\"missing\",\"arguments\":{}}</tool_call>"
                : "<think>Explain the result</think>Done", List.of(), 1);
        var original = agent.runTurn(List.of(), "Check", thinking::answer, thinking::reasoning);
        var displayed = thinking.finish(original);
        assertEquals("<think>Check first\n\nExplain the result</think>Done", displayed.reply());
        assertEquals(original.messages(), displayed.messages());
        assertEquals("Done", original.reply());
        assertTrue(updates.contains("<think>Check first</think>"));
        assertTrue(updates.stream().noneMatch(update -> update.contains("<tool_call>")));
    }

    @Test void handlesIncompleteProgressAndAResponseWithoutReasoning() {
        var updates = new ArrayList<String>();
        var thinking = new AgentThinking(updates::add);
        thinking.answer("Plain answer");
        assertEquals(List.of("Plain answer"), updates);
        thinking.reasoning(new Agent.Reasoning(0, "Working", false));
        assertEquals("<think>Working", updates.get(1));
        thinking.reasoning(new Agent.Reasoning(0, "Ready", true));
        assertEquals("<think>Ready</think>Plain answer", updates.get(2));
        var silent = new AgentThinking(null);
        silent.reasoning(new Agent.Reasoning(0, "Ready", true));
        assertEquals("<think>Ready</think>Done", silent.finish(Agent.Turn.answered("Test", "Done")).reply());
        assertEquals("Done", new AgentThinking(null).finish(Agent.Turn.answered("Test", "Done")).reply());
    }
}
