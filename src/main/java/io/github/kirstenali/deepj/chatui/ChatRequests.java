package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import io.github.kirstenali.deepj.agent.AgentHistory;
import javafx.concurrent.Task;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Stores the conversation snapshot that belongs to each original response. */
final class ChatRequests {
    private final ChatService service;
    private final AgentHistory history = new AgentHistory();
    private final Map<Integer, Request> requests = new HashMap<>();
    private boolean historyEnabled;

    ChatRequests(ChatService service) { this.service = service; }

    void useHistory(boolean enabled) { historyEnabled = enabled; }

    Request add(int response, String prompt, ChatGenerationSettings settings) {
        var request = new Request(prompt, settings, historyEnabled ? history.snapshot() : List.of());
        requests.put(response, request);
        return request;
    }

    Request remove(int response) { return requests.remove(response); }

    void complete(int response, Agent.Turn turn) {
        if (turn.reply() == null || turn.reply().isBlank()) remove(response);
        else if (service instanceof ConversationChatService) history.remember(turn);
    }

    void clear() {
        requests.clear();
        history.clear();
    }

    Task<Agent.Turn> generate(Request request, ChatGenerationSettings settings) {
        List<Agent.Turn> context = historyEnabled ? request.history() : List.of();
        return new Task<>() {
            @Override protected Agent.Turn call() {
                String prompt = request.prompt();
                return settings.generateTurn(service, prompt, context, this::updateMessage);
            }
        };
    }

    record Request(String prompt, ChatGenerationSettings settings, List<Agent.Turn> history) {}
}
