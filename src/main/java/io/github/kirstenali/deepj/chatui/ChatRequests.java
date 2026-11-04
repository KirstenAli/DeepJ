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

    ChatRequests(ChatService service) { this.service = service; }

    Request add(int response, String prompt, ChatGenerationSettings settings) {
        var request = new Request(prompt, settings, history.snapshot());
        requests.put(response, request);
        return request;
    }

    Request remove(int response) { return requests.remove(response); }

    void complete(int response, Agent.Turn turn) {
        if (turn.reply() == null || turn.reply().isBlank()) remove(response);
        else if (service instanceof AgentChatService) history.remember(turn);
    }

    void clear() {
        requests.clear();
        history.clear();
    }

    Task<Agent.Turn> generate(Request request, ChatGenerationSettings settings) {
        return new Task<>() {
            @Override protected Agent.Turn call() {
                String prompt = request.prompt();
                if (service instanceof AgentChatService agent) {
                    return agent.generateTurn(prompt, request.history(), settings, this::updateMessage);
                }
                return Agent.Turn.answered(prompt, settings.generate(service, prompt, this::updateMessage));
            }
        };
    }

    record Request(String prompt, ChatGenerationSettings settings, List<Agent.Turn> history) {}
}
