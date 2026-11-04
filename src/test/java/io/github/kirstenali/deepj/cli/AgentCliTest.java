package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.agent.AgentModel;
import io.github.kirstenali.deepj.agent.InteractiveTerminal;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class AgentCliTest extends CliTestSupport {
    @TempDir Path directory;

    @Test
    void modelsWithoutAnAdapterUseRegularChat() {
        console.inputs.addAll(List.of("Hello", "/exit"));
        assertEquals(0, AgentCli.execute(services("", true), "--color", "never", "model.dj"));
        assertTrue(output.toString().contains("Starting regular chat"));
        assertEquals(List.of("Hello"), model.prompts);
        assertTrue(model.closed);
    }

    @ParameterizedTest
    @ValueSource(strings = {"yes", ""})
    void runsTheCommandOnlyAfterAffirmativeApproval(String approval) throws Exception {
        assumeTrue(InteractiveTerminal.available());
        console.inputs.addAll(List.of("Use pwd", approval, "/exit"));
        var view = new CliView(console, new CliOptions(), Map.of("NO_COLOR", "1"));
        var agent = new AgentCliSession(console, view, scriptedModel(approval.equals("yes")), directory);
        agent.run();
        assertTrue(output.toString().contains("Requested command: /bin/pwd"));
        assertTrue(output.toString().contains("Working directory: " + directory));
        assertTrue(output.toString().contains("Done"));
    }

    private AgentModel scriptedModel(boolean allowed) throws Exception {
        String expected = allowed ? directory.toRealPath().toString() : "Command denied by terminal policy";
        return messages -> {
            if (messages.size() == 2) return """
                    <tool_call>{"name":"terminal","arguments":{"argv":["pwd"]}}</tool_call>
                    """;
            assertTrue(messages.get(3).content().contains(expected), messages.get(3).content());
            return "Done";
        };
    }
}
