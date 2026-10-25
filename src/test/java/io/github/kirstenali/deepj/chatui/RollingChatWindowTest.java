package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.models.ContextWindowMode;
import java.util.Collections;

import io.github.kirstenali.deepj.chatui.ChatWindowTestSupport.AlternativeService;
import io.github.kirstenali.deepj.chatui.ChatWindowTestSupport.TestService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class RollingChatWindowTest extends ChatDesktopTestSupport {
    @Test
    void rollingCheckboxReachesJavaThroughTheRealWebViewBridge() throws Exception {
        TestService service = new TestService();
        Fixture fixture = open(service);
        verifyWithoutClosing(fixture, view -> script(view, "setModel('stub.dj');"
                + "document.getElementById('rollingCache').checked = true;"
                + "document.getElementById('prompt').value = 'Hello'; submitPrompt()"));
        assertTrue(service.generated.await(10, TimeUnit.SECONDS));
        awaitScript(fixture, "!document.getElementById('prompt').disabled");
        assertEquals(ContextWindowMode.ROLLING, service.contextWindow);
        verify(fixture, view -> assertEquals(true, script(view, "document.getElementById('rollingCache').checked")));
    }

    @Test
    void desktopAlternativesReuseTheOriginalRollingChoice() throws Exception {
        AlternativeService service = new AlternativeService();
        Fixture fixture = open(service);
        verifyWithoutClosing(fixture, view -> script(view,
                "setModel('stub.dj'); chatBridge.sendWithWindowSettings('Hello', 80, 0.8, 40, 1234, 1, true)"));
        assertTrue(service.first.await(10, TimeUnit.SECONDS));
        awaitScript(fixture, "!!document.querySelector('.alternatives-button')");
        verifyWithoutClosing(fixture, view -> script(view, "document.querySelector('.alternatives-button').click()"));
        assertTrue(service.extra.await(10, TimeUnit.SECONDS));
        awaitScript(fixture, "!document.getElementById('prompt').disabled");
        assertEquals(Collections.nCopies(3, ContextWindowMode.ROLLING),
                service.modes);
        verify(fixture, view -> assertEquals(3, service.calls.size()));
    }
}
