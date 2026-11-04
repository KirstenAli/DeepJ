package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatFixture;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatModel;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

class Qwen3AgentModelTest {
    @Test
    void usesTheLoadedModelAndReturnsOnlyNewText(@TempDir Path directory) throws Exception {
        var chat = Qwen3ChatModel.load(Qwen3ChatFixture.create(directory.resolve("snapshot")));
        var agent = new Agent(new Qwen3AgentModel(chat, 4, 0.7f, 1, 1L), List.of(), 0);
        assertEquals("xxxx", agent.run("Hello"));
        assertThrows(IllegalStateException.class, () -> agent.run("x".repeat(256)));
    }

    @Test
    void dropsOversizedEarlierTurnsBeforeReducingTheCurrentAnswerBudget(@TempDir Path directory) throws Exception {
        var chat = Qwen3ChatModel.load(Qwen3ChatFixture.create(directory.resolve("snapshot")));
        var agent = new Agent(new Qwen3AgentModel(chat, 4, 0.7f, 1, 1L), List.of(), 0);
        var earlier = Agent.Turn.answered("Long previous turn", "x".repeat(256));
        assertEquals("xxxx", agent.runTurn(List.of(earlier), "Hello", null).reply());
    }
}
