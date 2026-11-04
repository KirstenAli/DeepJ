package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class ChatRequestsTest {
    @Test
    void alternativesUseTheOriginalSnapshotAndNewChatClearsIt() {
        var requests = new ChatRequests(new AgentChatService(command -> Optional.empty()));
        var settings = new ChatGenerationSettings(32, 0.8, 10, 123L, 1);
        var first = Agent.Turn.answered("file.txt", "I found it");
        var second = Agent.Turn.answered("Where is it?", "docs/file.txt");
        requests.add(0, "file.txt", settings);
        requests.complete(0, first);
        var original = requests.add(1, "Where is it?", settings);
        requests.complete(1, second);
        assertEquals(original, requests.remove(1));
        assertEquals(List.of(first), original.history());
        assertEquals(List.of(first, second), requests.add(2, "Can you open it?", settings).history());
        requests.clear();
        assertNull(requests.remove(0));
        assertEquals(List.of(), requests.add(3, "Where is it?", settings).history());
    }
}
