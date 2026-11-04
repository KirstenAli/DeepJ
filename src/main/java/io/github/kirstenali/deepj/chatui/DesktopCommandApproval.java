package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.agent.InteractiveTerminal.Request;
import javafx.application.Platform;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonBar.ButtonData;
import javafx.scene.control.ButtonType;
import javafx.stage.Modality;
import javafx.stage.Window;

import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicReference;

final class DesktopCommandApproval {
    private final Window owner;

    DesktopCommandApproval(Window owner) { this.owner = owner; }

    boolean approve(Request request) {
        var alert = new AtomicReference<Alert>();
        var task = new FutureTask<>(() -> show(request, alert));
        Platform.runLater(task);
        try {
            return task.get();
        } catch (InterruptedException stopped) {
            task.cancel(false);
            Platform.runLater(() -> close(alert.get()));
            Thread.currentThread().interrupt();
            throw new CancellationException("Agent stopped");
        } catch (ExecutionException failure) {
            throw new IllegalStateException("Could not display terminal approval", failure.getCause());
        }
    }

    private boolean show(Request request, AtomicReference<Alert> active) {
        if (!owner.isShowing()) return false;
        var run = new ButtonType("Run once", ButtonData.OK_DONE);
        var deny = new ButtonType("Deny", ButtonData.CANCEL_CLOSE);
        var alert = new Alert(Alert.AlertType.CONFIRMATION, "", run, deny);
        alert.initOwner(owner);
        alert.initModality(Modality.WINDOW_MODAL);
        alert.setTitle("DeepJ Agent");
        alert.setHeaderText("Allow this terminal command?");
        alert.setContentText(request.argv().get(0) + "\n\nWorking directory\n" + request.directory());
        ((Button) alert.getDialogPane().lookupButton(run)).setDefaultButton(false);
        ((Button) alert.getDialogPane().lookupButton(deny)).setDefaultButton(true);
        active.set(alert);
        return alert.showAndWait().orElse(deny) == run;
    }

    private static void close(Alert alert) {
        if (alert != null) alert.close();
    }
}
