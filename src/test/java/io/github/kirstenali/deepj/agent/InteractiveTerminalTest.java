package io.github.kirstenali.deepj.agent;

import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

@EnabledOnOs({OS.MAC, OS.LINUX})
class InteractiveTerminalTest {
    private static final String PWD_CALL = """
            <tool_call>
            {"name": "terminal", "arguments": {"command": ["pwd"]}}
            </tool_call>
            """.strip();

    @TempDir Path directory;

    @Test
    void resolvesPathInOrderAndAcceptsExplicitExecutables() throws Exception {
        Path first = executable("first");
        Path second = executable("second");
        String path = String.join(File.pathSeparator, first.getParent().toString(), second.getParent().toString());
        var lookup = TerminalCommands.local(directory, path).lookup();
        assertEquals(first, lookup.apply("probe"));
        assertEquals(second, lookup.apply(second.toString()));
        assertEquals(directory.resolve("./second/probe"), lookup.apply("./second/probe"));
        assertNull(lookup.apply("unknown"));
    }

    @Test
    void runsAnApprovedPipelineAndReturnsItsOutput() throws Exception {
        var approved = new AtomicReference<List<String>>();
        var tool = TerminalTool.local(directory, argv -> {
            approved.set(argv);
            return true;
        }, Duration.ofSeconds(5));
        List<String> argv = List.of("/bin/sh", "-c", "printf '%s' '</tool_call>' | /usr/bin/tr 'a-z' 'A-Z'");
        String payload = Qwen3Json.stringify(Map.of("name", "terminal", "arguments", Map.of("argv", argv)));
        var call = ToolProtocol.calls("<tool_call>" + payload + "</tool_call>").get(0);
        var result = tool.execute(call.arguments());
        assertEquals(argv, approved.get());
        assertEquals("</TOOL_CALL>", result.get("output"));
        assertEquals(0, result.get("exitCode"));
    }

    @Test
    void recognizesQwensTerminalCallAfterProseAndRequestsApproval() throws Exception {
        var approved = new AtomicReference<InteractiveTerminal.Request>();
        var terminal = new InteractiveTerminal(directory, request -> {
            approved.set(request);
            return true;
        });
        var agent = terminal.agent(messages -> {
            if (messages.size() == 2) return """
                    It seems there was an issue with the terminal command execution. Let me try running
                    the `pwd` command again to check the directory you're in.

                    """ + PWD_CALL;
            return commandOutput(messages.get(3).content());
        });
        assertEquals(directory.toRealPath().toString(), agent.run("What directory am I in?"));
        assertEquals(directory, approved.get().directory());
        assertEquals(1, approved.get().argv().size());
        Path executable = Path.of(approved.get().argv().get(0));
        assertTrue(executable.isAbsolute());
        assertEquals("pwd", executable.getFileName().toString());
    }

    @Test
    void displaysFencedToolCallExamplesWithoutRequestingApproval() throws Exception {
        String example = "Here is a tool call example.\n\n```xml\n" + PWD_CALL + "\n```";
        var terminal = new InteractiveTerminal(directory, request -> fail("Unexpected approval request"));
        assertEquals(example, terminal.agent(messages -> example).run("Show an example"));
    }

    private static String commandOutput(String response) {
        var payload = Qwen3Json.object(Qwen3Json.parse(response.lines().skip(1).findFirst().orElseThrow()));
        var result = Qwen3Json.object(payload.get("result"));
        assertEquals(0, Qwen3Json.integer(result.get("exitCode")));
        return Qwen3Json.string(result.get("output")).strip();
    }

    @Test
    void deniesACommandWithoutCreatingTheRequestedFile() throws Exception {
        var tool = TerminalTool.local(directory, argv -> false, Duration.ofSeconds(5));
        var result = tool.execute(Map.of("argv", List.of("/usr/bin/touch", "marker")));
        assertEquals("Command denied by terminal policy", result.get("error"));
        assertFalse(Files.exists(directory.resolve("marker")));
    }

    @Test
    void showsAllArgumentsAndEscapesControlCharactersInApproval() {
        var request = new InteractiveTerminal.Request(directory, List.of("/bin/sh", "-c", "one\ntwo\u202e"));
        String preview = request.preview();
        String command = "Command and arguments: [\"/bin/sh\", \"-c\", \"one\\u000atwo\\u202e\"]";
        assertEquals(command + "\nWorking directory: " + Qwen3Json.stringify(directory.toString()), preview);
    }

    private Path executable(String parent) throws Exception {
        Path path = Files.createDirectories(directory.resolve(parent)).resolve("probe");
        return Files.createSymbolicLink(path, Path.of("/bin/pwd"));
    }
}
