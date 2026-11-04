package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import java.util.List;
import java.util.function.Consumer;

interface ConversationChatService extends ChatService {
    Agent.Turn generateTurn(String prompt, List<Agent.Turn> history, ChatGenerationSettings settings,
                            Consumer<String> listener);
}
