package io.github.kirstenali.deepj.agent;

import java.util.List;
import java.util.function.Consumer;

/** Generates an assistant reply from a conversation using the agent's Hermes tool format. */
@FunctionalInterface
public interface AgentModel {
    String generate(List<Message> messages);

    /** Streams cumulative text when supported by the model. */
    default String stream(List<Message> messages, Consumer<String> listener) {
        return generate(messages);
    }

    enum Role { SYSTEM, USER, ASSISTANT, TOOL }

    record Message(Role role, String content) {}
}
