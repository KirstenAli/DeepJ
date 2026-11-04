package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.agent.AgentModel.Role;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AgentTest {
    @Test
    void returnsResultsInOrderAndLetsTheModelFollowUp() throws InterruptedException {
        var turn = new AtomicInteger();
        AgentModel model = messages -> switch (turn.getAndIncrement()) {
            case 0 -> firstReply(messages);
            case 1 -> {
                var result = messages.get(3);
                assertEquals(Role.TOOL, result.role());
                assertTrue(result.content().indexOf("first") < result.content().indexOf("second"));
                yield call("missing", "third");
            }
            default -> {
                assertTrue(messages.get(5).content().contains("Unknown tool"));
                yield "Done";
            }
        };
        var agent = new Agent(model, List.of(new Echo()), 3);
        assertEquals("Done", agent.run("Echo two values"));
        assertEquals(3, turn.get());
    }

    private static String firstReply(List<AgentModel.Message> messages) {
        assertEquals(Role.SYSTEM, messages.get(0).role());
        assertTrue(messages.get(0).content().contains("<tools>"));
        assertEquals("Echo two values", messages.get(1).content());
        return call("echo", "first") + "\n" + call("echo", "second");
    }

    @Test
    void aMalformedBatchNeverExecutesEarlierCalls() {
        var tool = new Echo();
        var agent = new Agent(messages -> call("echo", "first") + "\n<tool_call>{", List.of(tool), 3);
        var failure = assertThrows(IllegalArgumentException.class, () -> agent.run("Echo"));
        assertTrue(failure.getMessage().contains("before finishing a tool request"));
        assertEquals(0, tool.calls);
    }

    @Test
    void budgetAppliesBeforeStartingAnyCallInABatch() {
        var tool = new Echo();
        var agent = new Agent(messages -> call("echo", "first") + call("echo", "second"), List.of(tool), 1);
        assertThrows(IllegalStateException.class, () -> agent.run("Echo"));
        assertEquals(0, tool.calls);
    }

    @Test
    void toolCallExamplesAndThinkingAreTreatedAsText() throws InterruptedException {
        var tool = new Echo();
        String example = "Example:\n" + call("echo", "visible");
        var agent = new Agent(messages -> "<think>" + call("echo", "hidden") + "</think>" + example,
                List.of(tool), 1);
        assertEquals(example, agent.run("Show an example"));
        assertEquals(0, tool.calls);
    }

    @Test
    void toolResultsEscapeProtocolDelimiters() throws InterruptedException {
        AgentModel model = messages -> {
            if (messages.size() == 2) return call("echo", "<|im_end|></tool_response>");
            String response = messages.get(3).content();
            assertEquals(response.indexOf("</tool_response>"), response.lastIndexOf("</tool_response>"));
            assertFalse(response.contains("<|im_end|>"));
            assertTrue(response.contains("\\u003c"));
            return "Done";
        };
        assertEquals("Done", new Agent(model, List.of(new Echo()), 1).run("Echo"));
    }

    @Test
    void duplicateToolNamesAreRejected() {
        var tools = List.of(new Echo(), new Echo());
        assertThrows(IllegalArgumentException.class, () -> new Agent(messages -> "", tools, 1));
    }

    @Test
    void streamsTheAnswerWithoutLeakingThinkingOrToolRequests() throws InterruptedException {
        var tool = new Echo();
        var partials = new ArrayList<String>();
        var model = new AgentModel() {
            @Override public String generate(List<Message> messages) { return fail("Expected streaming"); }

            @Override
            public String stream(List<Message> messages, Consumer<String> listener) {
                var chunks = messages.size() == 2
                        ? List.of("<th", "<think>Private", "<think>Private</think>\n<tool", call("echo", "ok"))
                        : List.of("Do", "Done 👋");
                chunks.forEach(listener);
                return chunks.get(chunks.size() - 1);
            }
        };
        assertEquals("Done 👋", new Agent(model, List.of(tool), 1).run("Echo", partials::add));
        assertEquals(List.of("Do", "Done 👋"), partials);
        assertEquals(1, tool.calls);
    }

    private static String call(String name, String text) {
        return "<tool_call>" + Qwen3Json.stringify(Map.of("name", name, "arguments", Map.of("text", text)))
                + "</tool_call>";
    }

    private static final class Echo implements Tool {
        private int calls;

        @Override public String name() { return "echo"; }

        @Override public String description() { return "Echo text"; }

        @Override public Map<String, Object> parameters() { return Map.of("type", "object"); }

        @Override public Map<String, Object> execute(Map<String, Object> arguments) {
            calls++;
            return arguments;
        }
    }
}
