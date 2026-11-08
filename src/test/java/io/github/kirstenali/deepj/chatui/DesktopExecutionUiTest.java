package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.tensor.Tensor;
import java.util.concurrent.atomic.AtomicReference;
import netscape.javascript.JSObject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class DesktopExecutionUiTest extends ChatDesktopTestSupport {
    @Test
    void backendPreferenceUsesTheDesktopBridgeWithoutSwitchingTheLiveBackend() throws Exception {
        var value = new AtomicReference<>("auto");
        var preference = new DesktopBackendPreference(value::get, value::set);
        var settings = new DesktopExecution(preference, DesktopBackend.AUTO);
        var previous = Tensor.backend();
        var fixture = open();
        verify(fixture, view -> {
            ((JSObject) script(view, "window")).setMember("executionBridge", settings);
            script(view, "executionReady();openSettings();element('executionBackend').value='cpu';"
                    + "element('executionBackend').dispatchEvent(new Event('change'))");
            assertEquals("cpu", value.get());
            assertEquals("Restart DeepJ to use CPU.", script(view, "element('executionHelp').textContent"));
            assertSame(previous, Tensor.backend());
            assertTrue(styleNumber(view, "#executionBackend", "height") >= 36);
        });
    }
}
