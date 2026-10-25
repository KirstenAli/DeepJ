package io.github.kirstenali.deepj.cli;

import io.github.kirstenali.deepj.models.ContextWindowMode;

import org.junit.jupiter.api.Test;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class CliTextAndSettingsTest {

    @Test
    void rollingModeSurvivesEditsAndBothAlternativeSettings() {
        var settings = new SamplingSettings(80, 0.8f, 40, 7, 1).edit("context-window", "rolling");
        var rolling = ContextWindowMode.ROLLING;
        assertEquals(rolling, settings.contextWindow());
        assertEquals(rolling, settings.edit("seed", "9").contextWindow());
        assertEquals(rolling, settings.alternativePenalty().contextWindow());
        assertEquals(rolling, settings.alternativeSeed().contextWindow());
        assertThrows(IllegalArgumentException.class, () -> settings.edit("context-window", "invalid"));
        assertThrows(NullPointerException.class, () -> new SamplingSettings(80, 1, 0, 1, 1, null));
        var options = new SamplingOptions();
        options.rollingCache = true;
        assertEquals(rolling, options.settings().contextWindow());
    }
    @Test
    void streamingIsAppendOnlyAndWaitsForIncompleteUnicodeTails() {
        StringWriter text = new StringWriter();
        StreamedText stream = new StreamedText(new PrintWriter(text));
        stream.update("Hello\ufffd");
        assertEquals("Hello", text.toString());
        stream.update("Hello\ud83d");
        stream.update("Hello😀");
        stream.complete("Hello😀\n");
        assertEquals("Hello😀\n", text.toString());
        assertTrue(stream.endsWithNewline());
        assertFalse(stream.empty());
        assertThrows(IllegalStateException.class, () -> stream.update("different"));
    }

    @Test
    void streamingAndLabelsCannotInjectTerminalControls() {
        StringWriter text = new StringWriter();
        StreamedText stream = new StreamedText(new PrintWriter(text));
        assertTrue(stream.empty());
        stream.update("\u001b[31mhello\u0007");
        assertEquals("[31mhello", text.toString());
        assertEquals("hello world !", CliText.label("hello\nworld\t!"));
        Thread.currentThread().interrupt();
        try {
            stream.update("ignored");
            assertEquals("[31mhello", text.toString());
        } finally {
            Thread.interrupted();
        }
    }

    @Test
    void settingsEditsAndAlternativesPreserveAllUnchangedValues() {
        var initial = new SamplingSettings(80, 0.8f, 40, 7, 1);
        var edited = initial.edit("max-tokens", "12").edit("temperature", "0.5").edit("top-k", "0")
                .edit("seed", "42").edit("repetition-penalty", "1.2");
        assertEquals(new SamplingSettings(12, 0.5f, 0, 42, 1.2f), edited);
        assertEquals(7, initial.alternativePenalty().seed());
        assertEquals(8, initial.alternativeSeed().seed());
        assertEquals(1.1f, new SamplingSettings(1, 1, 0, 1, 2).alternativePenalty().repetitionPenalty());
        assertThrows(IllegalArgumentException.class, () -> initial.edit("unknown", "1"));
        assertThrows(IllegalArgumentException.class, () -> initial.edit("seed", "bad"));
    }

    @Test
    void rejectsInvalidSamplingValuesAndUnavailableForcedMetal() {
        assertThrows(IllegalArgumentException.class, () -> new SamplingSettings(0, 1, 0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new SamplingSettings(1, 0, 0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new SamplingSettings(1, Float.NaN, 0, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new SamplingSettings(1, 1, -1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new SamplingSettings(1, 1, 0, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new SamplingSettings(1, 1, 0, 1, Float.NaN));
        assertEquals(BackendChoice.CPU, BackendChoice.AUTO.resolve(() -> false));
        assertEquals(BackendChoice.METAL, BackendChoice.AUTO.resolve(() -> true));
        assertEquals(BackendChoice.CPU, BackendChoice.CPU.resolve(() -> true));
        assertThrows(IllegalArgumentException.class, () -> BackendChoice.METAL.resolve(() -> false));
    }

    @Test
    void bannerAdaptsToColorPreferencesAndTerminalSize() {
        CliOptions options = new CliOptions();
        assertTrue(style(options, true, 80, 256, Map.of()).banner().contains("\u001b[38;5;"));
        assertFalse(style(options, true, 80, 256, Map.of("NO_COLOR", "")).banner().contains("\u001b"));
        assertFalse(style(options, false, 80, 0, Map.of()).banner().contains("████"));
        assertFalse(style(options, true, 40, 8, Map.of()).banner().contains("████"));
        options.color = CliOptions.Color.NEVER;
        assertFalse(style(options, true, 80, 256, Map.of()).banner().contains("\u001b"));
        options.color = CliOptions.Color.ALWAYS;
        options.theme = CliTheme.MONO;
        assertTrue(style(options, true, 80, 8, Map.of()).banner().contains("\u001b[1m"));
        options.noBanner = true;
        assertEquals("", style(options, true, 80, 8, Map.of()).banner());
    }

    private CliStyle style(CliOptions options, boolean interactive, int width, int colors,
                             Map<String, String> environment) {
        return new CliStyle(options, interactive, width, colors, environment);
    }
}
