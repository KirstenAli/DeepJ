package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.InteractiveTerminal.Request;
import javafx.application.Platform;
import javafx.scene.web.WebView;
import javafx.stage.Window;
import netscape.javascript.JSObject;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

/** Connects a pending terminal request to the approval dialog inside the chat view. */
public final class DesktopCommandApproval {
    private final Window owner;
    private CompletableFuture<Boolean> active;

    DesktopCommandApproval(Window owner) { this.owner = owner; }

    boolean approve(Request request) {
        var decision = new CompletableFuture<Boolean>();
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

    private void show(Request request, CompletableFuture<Boolean> decision) {
        if (decision.isDone() || active != null || !owner.isShowing()) {
            decision.complete(false);
            return;
        }
        try {
            JSObject window = window();
            active = decision;
            window.setMember("approvalBridge", this);
            window.call("openApproval", request.commandPreview(), request.directoryPreview());
        } catch (RuntimeException failure) {
            active = null;
            decision.completeExceptionally(failure);
        }
    }

    public void decide(boolean allow) {
        var decision = active;
        if (decision == null) return;
        try {
            dismiss(decision);
            decision.complete(allow);
        } catch (RuntimeException failure) {
            decision.completeExceptionally(failure);
        }
    }

    private void dismiss(CompletableFuture<Boolean> decision) {
        if (active != decision) return;
        active = null;
        if (owner.isShowing()) window().call("closeApproval");
    }

    private JSObject window() {
        var view = (WebView) owner.getScene().getRoot();
        return (JSObject) view.getEngine().executeScript("window");
    }
}
