package io.github.kirstenali.deepj.chatui;

import javafx.stage.DirectoryChooser;
import javafx.stage.Stage;

import java.io.File;
import java.util.function.BooleanSupplier;

/** Desktop bridge for the chat composer's terminal controls. */
public final class AgentWorkspace {
    private final Stage stage;
    private final AgentChatService service;
    private final BooleanSupplier busy;

    AgentWorkspace(Stage stage, AgentChatService service, BooleanSupplier busy) {
        this.stage = stage;
        this.service = service;
        this.busy = busy;
    }

    String directoryLabel() {
        return service.supportsAgent() ? service.directory().toString() : "";
    }

    public void setEnabled(boolean enabled) {
        if (canEdit()) service.enable(enabled);
    }

    public String chooseDirectory() {
        if (!canEdit()) return directoryLabel();
        var chooser = new DirectoryChooser();
        chooser.setTitle("Choose the terminal working directory");
        File current = service.directory().toFile();
        if (current.isDirectory()) chooser.setInitialDirectory(current);
        File selected = chooser.showDialog(stage);
        if (selected != null) service.directory(selected.toPath());
        return directoryLabel();
    }

    private boolean canEdit() {
        return !busy.getAsBoolean() && service.supportsAgent();
    }
}
