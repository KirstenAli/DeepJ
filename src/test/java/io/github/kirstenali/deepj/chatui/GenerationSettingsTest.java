package io.github.kirstenali.deepj.chatui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GenerationSettingsTest {

    @Test
    void parsesValidSettings() {
        GenerationSettings settings = GenerationSettings.parse("120", "0.7", "20", "2026");

        assertEquals(120, settings.maxTokens());
        assertEquals(0.7f, settings.temperature());
        assertEquals(20, settings.topK());
        assertEquals(2026L, settings.seed());
    }

    @Test
    void rejectsInvalidSettings() {
        assertThrows(IllegalArgumentException.class,
                () -> GenerationSettings.parse("0", "0.7", "20", "2026"));
        assertThrows(IllegalArgumentException.class,
                () -> GenerationSettings.parse("120", "NaN", "20", "2026"));
        assertThrows(IllegalArgumentException.class,
                () -> GenerationSettings.parse("120", "0.7", "none", "2026"));
    }
}
