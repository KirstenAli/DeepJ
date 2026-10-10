package io.github.kirstenali.deepj.chatui;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ChatThemeTest {

    @Test
    void exposesReadableThemeNames() {
        assertEquals("Light", ChatTheme.LIGHT.toString());
        assertEquals("theme-dark", ChatTheme.DARK.styleClass());
        assertEquals("Midnight", ChatTheme.MIDNIGHT.toString());
    }
}
