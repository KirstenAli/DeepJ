package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.InteractiveTerminal.Request;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import javafx.application.Platform;
import javafx.scene.web.WebView;
import javafx.stage.Window;
import netscape.javascript.JSObject;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/** Connects a pending terminal request to the approval dialog inside the chat view. */
public final class DesktopCommandApproval {
    private final Window owner;
    private Pending active;

    DesktopCommandApproval(Window owner) { this.owner = owner; }

    Optional<Request> review(Request request) {
        var decision = new CompletableFuture<Optional<Request>>();
        Platform.runLater(() -> show(request, decision));
        try {
            return decision.get();
        } catch (InterruptedException stopped) {
            decision.cancel(false);
            Platform.runLater(() -> dismiss(decision));
            Thread.currentThread().interrupt();
            throw new CancellationException("Agent stopped");
        } catch (ExecutionException failure) {
            throw new IllegalStateException("Could not display terminal approval", failure.getCause());
        }
    }

    private void show(Request request, CompletableFuture<Optional<Request>> decision) {
        if (decision.isDone() || active != null || !owner.isShowing()) {
            decision.complete(Optional.empty());
            return;
        }
        try {
            JSObject window = window();
            active = new Pending(request, decision);
            window.setMember("approvalBridge", this);
            window.call("openApproval", request.commandPreview(), request.directoryPreview());
        } catch (RuntimeException failure) {
            active = null;
            decision.completeExceptionally(failure);
        }
    }

    public String reviewEdit(String json) {
        var pending = active;
        if (pending == null || pending.decision().isDone()) return editError("This request has expired.");
        try {
            var argv = Qwen3Json.array(Qwen3Json.parse(json)).stream().map(Qwen3Json::string).toList();
            var revised = pending.request().revise(argv);
            active = new Pending(revised, pending.decision());
            return Qwen3Json.stringify(Map.of("preview", revised.commandPreview()));
        } catch (IllegalArgumentException invalid) {
            return editError(invalid.getMessage());
        }
    }

    private static String editError(String message) {
        return Qwen3Json.stringify(Map.of("error", message));
    }

    public void decide(boolean allow) {
        var pending = active;
        if (pending == null) return;
        var decision = pending.decision();
        try {
            dismiss(decision);
            decision.complete(allow ? Optional.of(pending.request()) : Optional.empty());
        } catch (RuntimeException failure) {
            decision.completeExceptionally(failure);
        }
    }

    private void dismiss(CompletableFuture<Optional<Request>> decision) {
        if (active == null || active.decision() != decision) return;
        active = null;
        if (owner.isShowing()) window().call("closeApproval");
    }

    private JSObject window() {
        var view = (WebView) owner.getScene().getRoot();
        return (JSObject) view.getEngine().executeScript("window");
    }

    private record Pending(Request request, CompletableFuture<Optional<Request>> decision) {}
}
