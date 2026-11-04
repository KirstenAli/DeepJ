package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.agent.AgentModel.Message;
import io.github.kirstenali.deepj.agent.AgentModel.Role;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.StringJoiner;
import java.util.concurrent.CancellationException;

/** Runs a single user request, returning tool results to the model until it answers. */
public final class Agent {
    private final AgentModel model;
    private final Map<String, Tool> tools = new LinkedHashMap<>();
    private final String instructions;
    private final int maxToolCalls;

    public Agent(AgentModel model, List<? extends Tool> tools, int maxToolCalls) {
        this.model = Objects.requireNonNull(model, "model");
        if (maxToolCalls < 0) throw new IllegalArgumentException("maxToolCalls must be nonnegative");
        this.maxToolCalls = maxToolCalls;
        tools.forEach(this::register);
        instructions = ToolProtocol.instructions(this.tools);
    }

    public String run(String prompt) throws InterruptedException {
        var messages = new ArrayList<>(List.of(new Message(Role.SYSTEM, instructions),
                new Message(Role.USER, Objects.requireNonNull(prompt, "prompt"))));
        int remaining = maxToolCalls;
        while (true) {
            checkInterrupted();
            String reply = ToolProtocol.withoutThinking(model.generate(List.copyOf(messages)));
            var calls = ToolProtocol.calls(reply);
            if (calls.isEmpty()) return reply;
            if (calls.size() > remaining) throw new IllegalStateException("Agent tool call limit exceeded");
            remaining -= calls.size();
            messages.add(new Message(Role.ASSISTANT, reply));
            messages.add(new Message(Role.TOOL, execute(calls)));
        }
    }

    private void register(Tool tool) {
        String name = tool.name();
        if (!name.matches("[A-Za-z0-9_-]{1,64}")) throw new IllegalArgumentException("Invalid tool name: " + name);
        if (tools.putIfAbsent(name, tool) != null) throw new IllegalArgumentException("Duplicate tool: " + name);
    }

    private String execute(List<ToolProtocol.Call> calls) throws InterruptedException {
        var responses = new StringJoiner("\n");
        for (var call : calls) {
            checkInterrupted();
            responses.add(ToolProtocol.response(call.name(), execute(call)));
        }
        return responses.toString();
    }

    private Map<String, Object> execute(ToolProtocol.Call call) throws InterruptedException {
        Tool tool = tools.get(call.name());
        if (tool == null) return Map.of("error", "Unknown tool");
        try {
            return tool.execute(call.arguments());
        } catch (IOException | IllegalArgumentException failure) {
            return Map.of("error", Objects.toString(failure.getMessage(), "Tool failed"));
        }
    }

    private static void checkInterrupted() {
        if (Thread.currentThread().isInterrupted()) throw new CancellationException("Agent stopped");
    }
}
