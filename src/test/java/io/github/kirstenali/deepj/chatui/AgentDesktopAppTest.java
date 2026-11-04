package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.InteractiveTerminal;
import io.github.kirstenali.deepj.agent.InteractiveTerminal.Request;
import io.github.kirstenali.deepj.nn.transformer.qwen3.Qwen3ChatFixture;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import javafx.application.Platform;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

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
            assertFalse(approve(owner, "denyCommand"));
            assertTrue(approve(owner, "allowCommand"));
        } finally {
            close(owner);
        }
    }

    private boolean approve(Stage owner, String action) throws Exception {
        var request = new Request(directory, List.of("/bin/sh", "-c", "printf 'one two'"));
        WebView view = onFx(() -> (WebView) owner.getScene().getRoot());
        awaitCondition(view, "typeof openApproval === 'function'");
        var pending = CompletableFuture.supplyAsync(() -> new DesktopCommandApproval(owner).review(request));
        awaitCondition(view, "!element('approvalBackdrop').hidden");
        onFx(() -> {
            assertEquals(request.commandPreview(), script(view, "element('approvalCommand').textContent"));
            assertEquals(request.directoryPreview(), script(view, "element('approvalDirectory').textContent"));
            assertEquals("denyCommand", script(view, "document.activeElement.id"));
            script(view, "element('" + action + "').click()");
            assertEquals(true, script(view, "element('approvalBackdrop').hidden"));
            return null;
        });
        return pending.get(5, TimeUnit.SECONDS).isPresent();
    }

    @Test
    void runsTheReviewedEditAndReturnsItsArgumentsAndOutputToTheModel() throws Exception {
        assumeTrue(InteractiveTerminal.available());
        Stage owner = onFx(AgentDesktopAppTest::openAgent);
        try {
            WebView view = onFx(() -> (WebView) owner.getScene().getRoot());
            awaitCondition(view, "typeof openApproval === 'function'");
            var pending = startTestCommand(owner);
            try {
                awaitCondition(view, "!element('approvalBackdrop').hidden");
                onFx(() -> editAndRun(view));
                assertEquals("two words|", pending.get(5, TimeUnit.SECONDS));
            } finally {
                pending.cancel(true);
            }
        } finally {
            close(owner);
        }
    }

    private FutureTask<String> startTestCommand(Stage owner) {
        var pending = new FutureTask<>(() -> runEditedCommand(owner));
        var worker = new Thread(pending);
        worker.setDaemon(true);
        worker.start();
        return pending;
    }

    private String runEditedCommand(Stage owner) throws InterruptedException {
        var terminal = InteractiveTerminal.withCommandReview(directory, new DesktopCommandApproval(owner)::review);
        return terminal.agent(messages -> {
            if (messages.size() == 2) return """
                    <tool_call>
                    {"name":"terminal","arguments":{"argv":["/bin/pwd"]}}
                    </tool_call>
                    """;
            String response = messages.get(3).content();
            String json = response.lines().skip(1).findFirst().orElseThrow();
            var result = Qwen3Json.object(Qwen3Json.object(Qwen3Json.parse(json)).get("result"));
            assertEquals(List.of("/usr/bin/printf", "%s|%s", "two words", ""), result.get("argv"));
            return Qwen3Json.string(result.get("output"));
        }).run("Check the working directory");
    }

    private static Void editAndRun(WebView view) {
        script(view, """
                element('editCommand').click();
                element('commandExecutable').value = '/usr/bin/printf';
                addCommandArgument('%s|%s');
                addCommandArgument('two words');
                addCommandArgument('');
                element('reviewCommandEdit').click();
                """);
        assertEquals("[\"/usr/bin/printf\", \"%s|%s\", \"two words\", \"\"]",
                script(view, "element('approvalCommand').textContent"));
        assertEquals("denyCommand", script(view, "document.activeElement.id"));
        script(view, "element('allowCommand').click()");
        return null;
    }

    private static Stage openAgent() {
        var stage = new Stage();
        new AgentDesktopApp().start(stage);
        return stage;
    }

    private static void close(Stage owner) throws Exception {
        onFx(() -> {
            owner.close();
            return null;
        });
    }

    private static void awaitCondition(WebView view, String condition) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            if (onFx(() -> Boolean.TRUE.equals(script(view, condition)))) return;
            Thread.sleep(20);
        }
        throw new AssertionError("Timed out waiting for " + condition);
    }

    private static <T> T onFx(Callable<T> action) throws Exception {
        var task = new FutureTask<>(action);
        Platform.runLater(task);
        return task.get(5, TimeUnit.SECONDS);
    }
}
