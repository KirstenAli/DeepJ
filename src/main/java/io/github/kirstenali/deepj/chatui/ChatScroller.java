package io.github.kirstenali.deepj.chatui;

import javafx.application.Platform;
import javafx.scene.control.ScrollPane;
import javafx.scene.input.ScrollEvent;

final class ChatScroller {

    private final ScrollPane pane;
    private boolean following = true;
    private boolean pending;

    ChatScroller(ScrollPane pane) {
        this.pane = pane;
        pane.addEventFilter(ScrollEvent.SCROLL, event -> stopFollowing());
    }

    void followLatest() {
        following = true;
        pending = false;
        pane.setVvalue(1.0);
    }

    void requestBottom() {
        if (!following || pending) return;
        pending = true;
        Platform.runLater(this::scrollToBottom);
    }

    private void stopFollowing() {
        following = false;
        pending = false;
    }

    private void scrollToBottom() {
        if (!pending) return;
        pending = false;
        pane.setVvalue(1.0);
    }
}
