package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/** Combines agent reasoning for the shared desktop panel while retaining the original history messages. */
final class AgentThinking {
    private final Map<Integer, Agent.Reasoning> steps = new LinkedHashMap<>();
    private final Consumer<String> listener;
    private String answer = "";

    AgentThinking(Consumer<String> listener) { this.listener = listener; }

    void reasoning(Agent.Reasoning progress) {
        steps.put(progress.step(), progress);
        publish();
    }

    void answer(String text) {
        answer = text;
        publish();
    }

    Agent.Turn finish(Agent.Turn turn) {
        return new Agent.Turn(display(turn.reply()), turn.messages());
    }

    private void publish() {
        if (listener != null) listener.accept(display(answer));
    }

    private String display(String reply) {
        if (steps.isEmpty()) return reply;
        String text = steps.values().stream().map(Agent.Reasoning::text).collect(Collectors.joining("\n\n"));
        boolean complete = steps.values().stream().allMatch(Agent.Reasoning::complete);
        return "<think>" + text + (complete ? "</think>" + reply : "");
    }
}
