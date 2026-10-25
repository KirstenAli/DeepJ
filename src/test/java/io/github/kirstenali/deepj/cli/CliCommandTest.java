package io.github.kirstenali.deepj.cli;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CliCommandTest extends CliTestSupport {
    @Test
    void helpAlsoRespectsExplicitColorSettings() {
        assertEquals(0, DeepJCli.execute(services("", false), "chat", "--color", "always", "--help"));
        assertTrue(output.toString().contains("\u001b"));
        output.getBuffer().setLength(0);
        assertEquals(0, DeepJCli.execute(services("", true), "chat", "--color", "never", "--help"));
        assertFalse(output.toString().contains("\u001b"));
    }

    @Test
    void discoveryCommandsNeverLoadWeights() {
        var services = services("", false);
        for (String[] args : new String[][]{{}, {"--help"}, {"--version"}, {"doctor"}, {"inspect", "fake"}}) {
            assertEquals(0, DeepJCli.execute(services, args));
        }
        assertEquals(0, loads);
        assertTrue(output.toString().contains("DeepJ Prism"));
        assertTrue(output.toString().contains("CPU fallback"));
        assertTrue(output.toString().contains("1,234"));
    }

    @Test
    void generationPreservesPromptAndKeepsStdoutClean() {
        assertEquals(0, DeepJCli.execute(services("", false), "generate", "fake", "--backend", "cpu",
                "--prompt", "What is Java?", "--max-tokens", "12", "--seed", "7"));
        assertEquals("Hello!" + System.lineSeparator(), output.toString());
        assertEquals(java.util.List.of("What is Java?"), model.prompts);
        assertEquals(BackendChoice.CPU, selected);
        assertEquals(12, model.settings.get(0).maxTokens());
        assertEquals(7, model.settings.get(0).seed());
        assertTrue(model.closed);
    }

    @Test
    void generationReadsPipedInputAndReportsInvalidOptionsWithoutLoading() {
        var services = services("Capitalization\n", false);
        assertEquals(0, DeepJCli.execute(services, "generate", "fake"));
        assertEquals("Capitalization\n", model.prompts.get(0));
        assertEquals(2, DeepJCli.execute(services, "generate", "fake", "--temperature", "NaN"));
        assertEquals(1, loads);
        assertTrue(errors.toString().contains("temperature"));
        assertEquals(2, DeepJCli.execute(services, "unknown"));
    }

    @Test
    void interactiveChatSupportsCommandsAndAlternativesWithoutChangingTheOriginalSettings() {
        console.inputs.addAll(java.util.List.of("/alternatives", "/help", "/settings", "/set seed 42",
                "/set temperature 0.5", "Hello", "/alternatives", "/theme ember", "/clear", "/exit"));
        assertEquals(0, DeepJCli.execute(services("", true), "chat", "fake", "--color", "never"));
        assertEquals(java.util.List.of("Hello", "Hello", "Hello"), model.prompts);
        assertEquals(42, model.settings.get(0).seed());
        assertEquals(42, model.settings.get(1).seed());
        assertEquals(43, model.settings.get(2).seed());
        assertEquals(1.1f, model.settings.get(1).repetitionPenalty());
        assertTrue(console.cleared && console.closed && model.closed);
        assertFalse(output.toString().contains("\u001b"));
    }

    @Test
    void chatRecoversFromInvalidCommandsAndInputInterrupts() {
        console.inputs.add(new org.jline.reader.UserInterruptException(""));
        console.inputs.addAll(java.util.List.of(" ", "/missing", "/set top-k -2", "/theme bad", "/set",
                "/theme", "/set missing 3", "//literal", "/exit"));
        assertEquals(0, DeepJCli.execute(services("", true), "chat", "fake"));
        assertEquals(java.util.List.of("/literal"), model.prompts);
        assertTrue(output.toString().contains("Unknown command"));
        assertTrue(output.toString().contains("Choose aurora"));
    }

    @Test
    void stoppingGenerationRestoresInputAndDoesNotRunTheSecondAlternative() {
        model.cancel = true;
        console.inputs.addAll(java.util.List.of("Hello", "/alternatives", "/exit"));
        assertEquals(0, DeepJCli.execute(services("", true), "chat", "fake"));
        assertEquals(2, model.prompts.size());
        assertFalse(Thread.currentThread().isInterrupted());
        assertTrue(output.toString().contains("Stopped."));
        assertTrue(model.closed && console.closed);
    }

    @Test
    void interruptedCpuWorkerIsAStopButGenuineRuntimeErrorsAreNotHidden() {
        model.cancel = true;
        model.cancellation = new RuntimeException(new InterruptedException());
        console.inputs.addAll(java.util.List.of("Hello", "/exit"));
        assertEquals(0, DeepJCli.execute(services("", true), "chat", "fake"));
        assertTrue(output.toString().contains("Stopped."));
        assertFalse(Thread.currentThread().isInterrupted());
        model.cancel = false;
        model.failure = new RuntimeException("broken model");
        console.inputs.add("Hello");
        assertEquals(1, DeepJCli.execute(services("", true), "chat", "fake"));
        assertTrue(errors.toString().contains("broken model"));
    }
}
