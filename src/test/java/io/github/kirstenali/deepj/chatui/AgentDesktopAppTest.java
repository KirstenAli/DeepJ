package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.InteractiveTerminal;
import io.github.kirstenali.deepj.agent.InteractiveTerminal.Request;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatFixture;
import javafx.application.Platform;
import javafx.scene.control.Button;
import javafx.scene.control.DialogPane;
import javafx.scene.control.TextArea;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import javafx.stage.Window;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "deepj.uiTests", matches = "true")
class AgentDesktopAppTest extends ChatDesktopTestSupport {
    @TempDir Path directory;

    @Test
    void launchesWithTheChatWebViewFillingTheWindow() throws Exception {
        onFx(() -> {
            Stage stage = openAgent();
            try {
                assertEquals("DeepJ Agent", stage.getTitle());
                assertInstanceOf(WebView.class, stage.getScene().getRoot());
            } finally {
                stage.close();
            }
            return null;
        });
    }

    @Test
    void placesAvailableAgentToolsInTheComposer() throws Exception {
        var service = new AgentChatService(request -> fail("Unexpected command"));
        service.directory(directory);
        service.loadModel(Qwen3ChatFixture.create(directory.resolve("qwen")));
        var fixture = open(service);
        verify(fixture, view -> {
            assertEquals(!InteractiveTerminal.available(), script(view, "element('agentControls').hidden"));
            assertEquals(true, script(view, "element('agentMenu').hidden"));
            assertEquals(false, script(view, "element('agentMode').checked"));
            if (!InteractiveTerminal.available()) return;
            assertEquals(directory.toString(), script(view, "element('agentDirectory').title"));
            script(view, "element('agentPicker').click(); element('agentMode').click()");
            assertEquals("Agent on", script(view, "element('agentLabel').textContent"));
            assertTrue(styleNumber(view, "#agentMenu", "width") >= 280);
        });
    }

    @Test
    void displaysTheExactCommandAndReturnsTheSelectedDecision() throws Exception {
        Stage owner = onFx(AgentDesktopAppTest::openAgent);
        try {
            assertFalse(approve(owner, "Deny"));
            assertTrue(approve(owner, "Run once"));
        } finally {
            onFx(() -> {
                owner.close();
                return null;
            });
        }
    }

    private boolean approve(Stage owner, String action) throws Exception {
        var request = new Request(directory, List.of("/bin/sh", "-c", "printf 'one two'"));
        var pending = CompletableFuture.supplyAsync(() -> new DesktopCommandApproval(owner).approve(request));
        DialogPane dialog = awaitDialog();
        onFx(() -> {
            var preview = assertInstanceOf(TextArea.class, dialog.getContent());
            assertFalse(preview.isEditable());
            assertEquals(request.preview(), preview.getText());
            assertTrue(button(dialog, "Deny").isDefaultButton());
            assertFalse(button(dialog, "Run once").isDefaultButton());
            button(dialog, action).fire();
            return null;
        });
        return pending.get(5, TimeUnit.SECONDS);
    }

    private static Button button(DialogPane dialog, String label) {
        var type = dialog.getButtonTypes().stream().filter(item -> item.getText().equals(label)).findFirst();
        return (Button) dialog.lookupButton(type.orElseThrow());
    }

    private static Stage openAgent() {
        var stage = new Stage();
        new AgentDesktopApp().start(stage);
        return stage;
    }

    private static DialogPane awaitDialog() throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            DialogPane dialog = onFx(AgentDesktopAppTest::findDialog);
            if (dialog != null) return dialog;
            Thread.sleep(20);
        }
        throw new AssertionError("Terminal approval dialog did not appear");
    }

    private static DialogPane findDialog() {
        return Window.getWindows().stream().filter(Window::isShowing)
                .map(window -> window.getScene().getRoot()).filter(DialogPane.class::isInstance)
                .map(DialogPane.class::cast).findFirst().orElse(null);
    }

    private static <T> T onFx(Callable<T> action) throws Exception {
        var task = new FutureTask<>(action);
        Platform.runLater(task);
        return task.get(5, TimeUnit.SECONDS);
    }
}
