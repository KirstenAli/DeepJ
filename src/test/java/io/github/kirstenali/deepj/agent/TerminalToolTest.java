package io.github.kirstenali.deepj.agent;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.*;

@EnabledOnOs({OS.MAC, OS.LINUX})
class TerminalToolTest {
    @TempDir Path workingDirectory;

    @Test
    void usesTheFixedWorkingDirectory() throws Exception {
        var tool = terminal("pwd", "/bin/pwd", argv -> argv.equals(List.of("/bin/pwd")));
        var result = tool.execute(arguments("pwd"));
        assertEquals(0, result.get("exitCode"));
        assertEquals(workingDirectory.toRealPath().toString(), ((String) result.get("output")).strip());
        assertEquals(false, result.get("truncated"));
    }

    @Test
    void checksArgumentsAndRejectsUnavailableCommands() throws Exception {
        var tool = terminal("pwd", "/bin/pwd", argv -> argv.equals(List.of("/bin/pwd")));
        assertEquals("Command denied by terminal policy", tool.execute(arguments("pwd", "-P")).get("error"));
        assertEquals("Command is not available", tool.execute(arguments("sh", "-c", "pwd")).get("error"));
        assertThrows(IllegalArgumentException.class, () -> tool.execute(Map.of("command", "pwd")));
        assertThrows(IllegalArgumentException.class, () -> tool.execute(arguments()));
        assertThrows(IllegalArgumentException.class, () -> tool.execute(arguments("pwd", "a\u0000b")));
    }

    @Test
    void treatsShellMetacharactersAsLiteralArguments() throws Exception {
        var tool = terminal("echo", "/bin/echo", argv -> argv.size() == 2);
        assertEquals("$(pwd); echo hi\n", tool.execute(arguments("echo", "$(pwd); echo hi")).get("output"));
    }

    @Test
    void returnsFailureExitCodeAndStderr() throws Exception {
        var tool = terminal("ls", "/bin/ls", argv -> argv.size() == 2);
        var result = tool.execute(arguments("ls", "missing-file"));
        assertNotEquals(0, result.get("exitCode"));
        assertTrue(((String) result.get("output")).contains("missing-file"));
    }

    @Test
    void boundsCapturedOutput() throws Exception {
        var tool = terminal("echo", "/bin/echo", argv -> argv.size() == 2);
        var result = tool.execute(arguments("echo", "a".repeat(32_768)));
        assertEquals(0, result.get("exitCode"));
        assertEquals(true, result.get("truncated"));
        assertEquals("a".repeat(8192), result.get("output"));
    }

    @Test
    void returnsAfterTheCommandTimesOut() {
        var tool = new TerminalTool(workingDirectory, Map.of("sleep", Path.of("/bin/sleep")),
                argv -> argv.equals(List.of("/bin/sleep", "30")), Duration.ofMillis(100));
        var result = assertTimeoutPreemptively(Duration.ofSeconds(3), () -> tool.execute(arguments("sleep", "30")));
        assertEquals("Command timed out", result.get("error"));
    }

    private TerminalTool terminal(String name, String path, Predicate<List<String>> authorize) {
        return new TerminalTool(workingDirectory, Map.of(name, Path.of(path)), authorize, Duration.ofSeconds(5));
    }

    private static Map<String, Object> arguments(String... argv) {
        return Map.of("argv", List.of(argv));
    }
}
