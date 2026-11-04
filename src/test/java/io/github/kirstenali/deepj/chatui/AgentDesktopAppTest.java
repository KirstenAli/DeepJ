package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.InteractiveTerminal.Request;
import javafx.application.Platform;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.DialogPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.ToolBar;
import javafx.scene.layout.BorderPane;
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
    void launchesTheChatWindowWithAgentModeDisabledUntilModelLoad() throws Exception {
        onFx(() -> {
            Stage stage = openAgent();
            try {
                assertEquals("DeepJ Agent", stage.getTitle());
                var layout = (BorderPane) stage.getScene().getRoot();
                var toolbar = (ToolBar) layout.getTop();
                var mode = (CheckBox) toolbar.getItems().get(0);
                assertEquals("Agent mode", mode.getText());
                assertTrue(mode.isDisabled());
                assertInstanceOf(WebView.class, layout.getCenter());
            } finally {
                stage.close();
            }
            return null;
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
