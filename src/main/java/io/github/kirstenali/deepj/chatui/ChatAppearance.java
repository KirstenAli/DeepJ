package io.github.kirstenali.deepj.chatui;

import javafx.scene.Parent;
import javafx.scene.control.Button;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;

import java.util.Map;

final class ChatAppearance {

    private final Parent root;
    private final VBox panel;
    private final Region backdrop;
    private final Map<ChatTheme, Button> buttons;

    ChatAppearance(Parent root, VBox panel, Region backdrop,
                   Button light, Button dark, Button midnight) {
        this.root = root;
        this.panel = panel;
        this.backdrop = backdrop;
        buttons = Map.of(ChatTheme.LIGHT, light, ChatTheme.DARK, dark,
                ChatTheme.MIDNIGHT, midnight);
    }

    void apply(ChatTheme theme) {
        theme.apply(root);
        buttons.forEach((value, button) -> select(button, value == theme));
    }

    void toggle() {
        setVisible(!panel.isVisible());
    }

    void hide() {
        setVisible(false);
    }

    private void setVisible(boolean visible) {
        panel.setManaged(visible);
        panel.setVisible(visible);
        backdrop.setManaged(visible);
        backdrop.setVisible(visible);
    }

    private static void select(Button button, boolean selected) {
        button.getStyleClass().remove("selected");
        if (selected) button.getStyleClass().add("selected");
    }
}
