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
        console.inputs.addAll(List.of("Print hello", approval, "What was the result?", "/exit"));
        var view = new CliView(console, new CliOptions(), Map.of("NO_COLOR", "1"));
        var agent = new AgentCliSession(console, view, scriptedModel(approval.equals("yes")), directory);
        agent.run();
        String command = "Command and arguments: [\"/bin/sh\", \"-c\", \"printf 'hello world'\"]";
        assertTrue(output.toString().contains(command));
        assertTrue(output.toString().contains("Working directory: \"" + directory + "\""));
        assertTrue(output.toString().contains("Done"));
        assertTrue(output.toString().contains(approval.equals("yes") ? "Earlier: hello world" : "Earlier: denied"));
    }

    @Test
    void pipedInputCannotApproveACommand() {
        assumeTrue(InteractiveTerminal.available());
        var services = services("Print hello\n/exit\n", false);
        var plain = new CliConsole.Plain(services);
        var view = new CliView(plain, new CliOptions(), Map.of("NO_COLOR", "1"));
        new AgentCliSession(plain, view, scriptedModel(false), directory).run();
        assertTrue(output.toString().contains("Approval requires an interactive terminal"));
        assertFalse(output.toString().contains("Run once?"));
    }

    private static AgentModel scriptedModel(boolean allowed) {
        String expected = allowed ? "hello world" : "Command denied by terminal policy";
        return messages -> {
            if (messages.size() == 2) return """
                    <tool_call>
                    {"name":"terminal","arguments":{"argv":["/bin/sh","-c","printf 'hello world'"]}}
                    </tool_call>
                    """;
            assertTrue(messages.get(3).content().contains(expected), messages.get(3).content());
            if (messages.size() == 6) {
                assertEquals("What was the result?", messages.get(5).content());
                return "Earlier: " + (allowed ? "hello world" : "denied");
            }
            return "Done";
        };
    }
}
