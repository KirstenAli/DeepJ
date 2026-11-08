package io.github.kirstenali.deepj.cli;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CliThinkingTest extends CliTestSupport {
    @Test void thinkingIsOptInAndSurvivesSamplingEditsAndAlternatives() {
        var initial = new SamplingOptions().settings();
        assertFalse(initial.thinking());
        var thinking = initial.edit("thinking", "ON").edit("seed", "7").edit("context-window", "rolling");
        assertTrue(thinking.thinking());
        assertTrue(thinking.alternativePenalty().thinking());
        assertTrue(thinking.alternativeSeed().thinking());
        assertFalse(thinking.edit("thinking", "off").thinking());
        assertThrows(IllegalArgumentException.class, () -> initial.edit("thinking", "maybe"));
    }

    @Test void chatFlagAndInteractiveToggleReachTheNextGeneration() {
        model.canThink = true;
        console.inputs.addAll(List.of("First", "/set seed 7", "Second", "/set thinking off", "Third",
                "/set thinking on", "/alternatives", "Fourth", "/exit"));
        assertEquals(0, DeepJCli.execute(services("", true), "chat", "fake", "--thinking"));
        assertEquals(List.of(true, true, false, false, false, true),
                model.settings.stream().map(SamplingSettings::thinking).toList());
        assertTrue(output.toString().contains("thinking"));
        assertEquals(7, model.settings.get(1).seed());
    }

    @Test void unsupportedModelsRejectThinkingAndRetainTheirUsualSettings() {
        assertEquals(2, DeepJCli.execute(services("", true), "chat", "fake", "--thinking"));
        assertTrue(model.closed);
        assertTrue(model.prompts.isEmpty());
        assertTrue(errors.toString().contains("Qwen3 only"));
        console.inputs.addAll(List.of("/set thinking on", "Hello", "/exit"));
        assertEquals(0, DeepJCli.execute(services("", true), "chat", "fake"));
        assertFalse(model.settings.get(0).thinking());
    }

    @Test void generateSupportsThinkingAndRejectsUnsupportedModels() {
        assertEquals(2, DeepJCli.execute(services("", false), "generate", "fake",
                "--prompt", "Hello", "--thinking"));
        model.canThink = true;
        assertEquals(0, DeepJCli.execute(services("", false), "generate", "fake",
                "--prompt", "Hello", "--thinking"));
        assertTrue(model.settings.get(0).thinking());
        assertEquals("Hello!" + System.lineSeparator(), output.toString());
    }

    @Test void everyThinkingMarkerSplitStreamsWithoutLeakingTagsOrRevisingOutput() {
        var text = new StringWriter();
        var stream = new StreamedText(new PrintWriter(text), true);
        String reply = "<think>Check 17 × 23. 😀\n</think>391";
        for (int end = 0; end <= reply.length(); end++) {
            stream.update(reply.substring(0, end));
        }
        stream.complete(reply);
        assertEquals("Thinking\nCheck 17 × 23. 😀\n\nAnswer\n391", text.toString());
        assertFalse(stream.empty());
        assertFalse(stream.endsWithNewline());
    }

    @Test void unfinishedReasoningIsLabelledAndOrdinaryTextRemainsOrdinary() {
        var text = new StringWriter();
        var stream = new StreamedText(new PrintWriter(text), true);
        stream.update("<think>Check the answer</thi");
        stream.complete("<think>Check the answer</thi");
        assertTrue(text.toString().contains("Thinking ended before an answer"));
        assertFalse(text.toString().contains("</thi"));
        assertEquals("Plain answer", ThinkingText.display("Plain answer"));
        assertEquals("", ThinkingText.display("<thi"));
        assertEquals("", ThinkingText.completionNote("Plain answer"));
    }
}
