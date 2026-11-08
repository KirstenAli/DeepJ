package io.github.kirstenali.deepj.chatui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import static org.junit.jupiter.api.Assertions.assertEquals;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class ComposerFocusTest extends ChatDesktopTestSupport {

    @Test
    void clickingDisabledComposerDoesNotLoseFocusAfterLoading() throws Exception {
        Fixture fixture = open();
        verify(fixture, view -> {
            view.requestFocus();
            script(view, "promptBox.value = 'A draft'; promptBox.focus(); promptBox.setSelectionRange(3, 3)");
            script(view, "setBusy(true)");
            assertEquals(false, script(view, "document.activeElement === promptBox"));
            script(view, "promptBox.focus(); setModel('Qwen3'); setBusy(false)");
            assertEquals(false, script(view, "promptBox.disabled"));
            assertEquals(true, script(view, "document.activeElement === promptBox"));
            assertEquals("A draft", script(view, "promptBox.value"));
            assertEquals(3, ((Number) script(view, "promptBox.selectionStart")).intValue());
        });
    }

    @Test
    void mediaLoadingRestoresTheComposerWhenNoOtherControlHasFocus() throws Exception {
        Fixture fixture = open();
        verify(fixture, view -> {
            script(view, "setBusy(true); setImageModel('SANA-Sprint'); setBusy(false)");
            assertEquals(true, script(view, "document.activeElement === promptBox"));
            script(view, "setBusy(true); setVideoModel('SANA-Video'); setBusy(false)");
            assertEquals(true, script(view, "document.activeElement === promptBox"));
        });
    }

    @Test
    void loadingDoesNotStealFocusFromSettings() throws Exception {
        Fixture fixture = open();
        verify(fixture, view -> {
            script(view, "setBusy(true); openSettings(); setModel('Qwen3'); setBusy(false)");
            assertEquals(false, script(view, "promptBox.disabled"));
            assertEquals("closeSettings", script(view, "document.activeElement.id"));
        });
    }

}
