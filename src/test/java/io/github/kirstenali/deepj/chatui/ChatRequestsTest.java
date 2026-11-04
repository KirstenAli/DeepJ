package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.Agent;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class ChatRequestsTest {
    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void alternativesUseTheOriginalSnapshotAndNewChatClearsIt(boolean agent) {
        var requests = requests(agent);
        requests.useHistory(true);
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

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void historyIsOptInForEachDesktopLauncher(boolean agent) {
        var requests = requests(agent);
        var settings = new ChatGenerationSettings(32, 0.8, 10, 123L, 1);
        var first = Agent.Turn.answered("file.txt", "I found it");
        requests.add(0, "file.txt", settings);
        requests.complete(0, first);
        assertEquals(List.of(), requests.add(1, "Where is it?", settings).history());
        requests.useHistory(true);
        assertEquals(List.of(first), requests.add(2, "Where is it?", settings).history());
        requests.useHistory(false);
        assertEquals(List.of(), requests.add(3, "Where is it?", settings).history());
    }

    private static ChatRequests requests(boolean agent) {
        return new ChatRequests(agent ? new AgentChatService(command -> Optional.empty()) : new DeepJChatService());
    }
}
