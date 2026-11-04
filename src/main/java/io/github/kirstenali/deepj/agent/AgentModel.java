package io.github.kirstenali.deepj.agent;

import java.util.List;

/** Generates an assistant reply from a conversation using the agent's Hermes tool format. */
@FunctionalInterface
public interface AgentModel {
    String generate(List<Message> messages);

    enum Role { SYSTEM, USER, ASSISTANT, TOOL }

    record Message(Role role, String content) {}
}
